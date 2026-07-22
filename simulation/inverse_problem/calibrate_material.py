"""
Direct inverse calibration: recover (E, nu) for a sim by optimizing the repo's own
forward MPM solver to match a recorded ground-truth trajectory (GtX.pt).

No NCLaw, no external checkpoint, no learned latent — this is derivative-free
optimization treating mpm_simulator_perparticleparams.py as a black box.

Usage (relative --sim_dir/--output_json paths are resolved against your cwd, so `cd
simulation/data_process` first to match the example below -- this script doesn't use hydra
and resolves its sibling import in data_process/ from its own file location, not cwd):
    cd simulation/data_process
    python ../inverse_problem/calibrate_material.py \
        --sim_dir ../output/001a_elastic_sphere_tissue_free_fall_test_soft \
        --output_json ../inverse_problem/results/001a_calibration.json

Verified end-to-end on 001a (2026-07-22, Titan X, 12GB, uniphy conda env):
`--sim_dir 001a --num_frames_opt 4 --max_iters 15` completed cleanly in ~11-12 min wall-clock
(34 objective evaluations + 1 full-length validation rollout, ~20s/evaluation average) with
zero CUDA_ERROR_ILLEGAL_ADDRESS, zero CFL violations, and no GPU memory growth trend across
iterations (oscillates between the ti.init() reservation, ~7GB at device_memory_fraction=0.5
of 12GB, and <1GB during each ti.reset() teardown -- returns to the same baseline every time,
not the same as a leak). Recovered E=15697 vs. true E=14999 (~5% off) -- reasonable given the
Nelder-Mead search only ran a handful of evaluations against a noisy objective (see below).
particles_ti_root read from config.yaml matched what MPMWrapper's constructor expects with no
adjustment needed.

Known limitations found from this run, not fixed here (would need a design decision, not a
one-line patch):
- **nu is unidentifiable at num_frames_opt=4**: every one of the 34 logged evaluations shows
  nu pinned at ~0.2500 (its initial value from x0), never moving toward the true 0.4500 --
  because 001a's floor contact doesn't start until ~frame 7 (floor_contact_time_s=0.297s at
  24fps), a 4-frame window is pure free-fall, which has no sensitivity to Poisson's ratio at
  all (nu only shows up once the sphere actually deforms on contact). Recovering nu needs
  --num_frames_opt large enough to include contact + some post-contact deformation, not just a
  short free-fall window -- try 8-10 for this sim specifically.
- **The loss floor looks noisy across the explored E range**: loss only varies ~0.0062-0.0070
  while E ranges over ~8,000-31,000 Pa (a ~4x spread) in the logged evaluations. Likely cause:
  `initialize_particles()` (mpmwrapper_perparticleparams.py) re-samples a fresh random subset
  of particles from the SDF point cloud via `np.random.choice(..., num_particles)` -- no fixed
  seed, no `replace=False` -- on *every* call, so every candidate rollout (including repeat
  evaluations at the same E/nu) starts from a different particle sample than the one GtX.pt was
  originally recorded with. This is pre-existing pipeline behavior (matches how the ground
  truth itself was generated, see CLAUDE.md's CFL-bisection seed-sensitivity note), not
  something introduced here, but it does put a noise floor under this optimizer's objective
  that a future revision might want to control for (e.g. seeding numpy before each
  initialize_particles() call, or averaging loss over a few resamples per candidate).
- **The CFL-violated penalty path (1e6) was never exercised in this run**: E stayed in the
  8,000-31,000 Pa range throughout, well under the ~3-8MPa range where CLAUDE.md documents CFL
  instability for this material/timestep. Whether the optimizer actually steers away from that
  penalty (vs. getting stuck) is untested -- would need a run with --init_E pushed much higher,
  or wider Nelder-Mead exploration, to provoke it.
- Fixed since the first draft: the sibling import of mpmwrapper_perparticleparams.py did not
  actually resolve when invoked as documented (`sys.path` was pointed at this script's own
  directory, not data_process/) -- now resolves relative to this file's location regardless of
  cwd. Also added line-buffered stdout (`sys.stdout.reconfigure(line_buffering=True)`) --
  without it, none of the per-iteration print()s appear until the whole run exits once stdout
  isn't a TTY, which defeats watching iteration-by-iteration progress live exactly as intended.
"""
import argparse
import json
import os
import sys
import copy
import time

import numpy as np
import torch
import taichi as ti
from omegaconf import OmegaConf
from scipy.optimize import minimize

# Once stdout isn't a TTY (redirected to a file/log, or captured by a background-task runner),
# Python fully buffers it instead of line-buffering -- every print() below (including the
# per-iteration "E=... nu=... loss=..." lines the whole point of this script is to let you
# watch live) would otherwise sit invisible until the entire run exits. Force line buffering.
sys.stdout.reconfigure(line_buffering=True)

# `mpmwrapper_perparticleparams.py` lives in data_process/, a sibling of this script's own
# inverse_problem/ directory -- NOT the script's own directory (plain `python script.py`
# puts the script's dir on sys.path[0], not the cwd, so running from data_process/ alone
# does not make this import resolve without this).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_process"))
from mpmwrapper_perparticleparams import MPMWrapper


TI_MEM_FRACTION = 0.5  # matches dataset_generate.py


def lame_from_E_nu(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def E_nu_from_lame(mu, lam):
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return E, nu


def load_ground_truth(sim_dir):
    cfg = OmegaConf.load(os.path.join(sim_dir, "config.yaml"))
    gt_x = torch.load(os.path.join(sim_dir, "GtX.pt"), map_location="cpu")
    gt_v = torch.load(os.path.join(sim_dir, "GtV.pt"), map_location="cpu")

    inv_dt = cfg["objects"]["sphere"]["material"]["inv_dt"]
    inv_frame_dt = cfg["objects"]["sphere"]["material"]["inv_frame_dt"]
    n_substeps = round(inv_dt / inv_frame_dt)
    num_frames = cfg["visualization_cfg"]["num_frames"]

    valid_len = num_frames * n_substeps  # trims the +2 unwritten tail
    gt_x = gt_x[:valid_len]
    gt_v = gt_v[:valid_len]

    return cfg, gt_x, gt_v, n_substeps


def run_forward(cfg, mu, lam, num_frames, gt_x0=None, gt_v0=None, verbose_timing=False):
    """Runs the repo's own solver with candidate (mu, lam), same loop shape as
    dataset_generate.py's main(), and returns the trimmed particle-position trajectory.

    gt_x0/gt_v0 (numpy float32 arrays, shape (num_particles, 3)), if given, seed the particle
    positions/velocities directly from the ground truth's recorded t=0 state instead of letting
    MPMWrapper.__init__ draw a fresh random particle subset from the SDF every call -- see the
    module docstring's "loss floor looks noisy" note on why that resampling matters. Safe
    specifically at t=0 (not pre-deformed state in general, which CLAUDE.md flags as unsafe for
    from_torch): from_torch always writes F=identity and C=zero into the slot it initializes,
    which is exactly correct before any physics substep has run.
    """
    cfg = copy.deepcopy(cfg)
    cfg["objects"]["sphere"]["material"]["mu"] = float(mu)
    cfg["objects"]["sphere"]["material"]["lam"] = float(lam)

    particles_ti_root = cfg["train_cfg"]["particles_ti_root"]
    cuda_chunk_size = cfg["train_cfg"]["cuda_chunk_size"]

    t_setup_start = time.time()
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=TI_MEM_FRACTION, debug=False)

    wrapper = MPMWrapper(cfg["objects"], cfg["simulator_cfg"],
                          particles_ti_root=particles_ti_root, cuda_chunk_size=cuda_chunk_size)
    wrapper.initialize_particles()
    if gt_x0 is not None:
        wrapper.init_particles = gt_x0
        wrapper.init_velocities = gt_v0 if gt_v0 is not None else wrapper.init_velocities
    wrapper.simulator_variables_initialize()
    t_setup = time.time() - t_setup_start

    n_substeps = wrapper.simulator.n_substeps[None]
    buffer_len = num_frames * n_substeps + 2
    device = "cuda"

    particle_x = torch.zeros([particles_ti_root, buffer_len, 3], dtype=torch.float32, device=device)

    t_physics_start = time.time()
    cfl_ok = True
    for f in range(num_frames):
        for i in range(n_substeps * f, n_substeps * (f + 1)):
            if wrapper.simulator.cfl_satisfy[None]:
                wrapper.simulator.substep(i)
            else:
                wrapper.simulator.cached_states.clear()
                cfl_ok = False
                break
            wrapper.simulator.get_x_gt(i, particle_x)
        if not cfl_ok:
            break
    t_physics = time.time() - t_physics_start

    num_particles = wrapper.num_particles[None]
    valid_len = num_frames * n_substeps
    pred_x = particle_x[:num_particles, :valid_len].permute(1, 0, 2).cpu()

    if verbose_timing:
        print(f"    [run_forward num_frames={num_frames}] setup={t_setup:.2f}s physics={t_physics:.2f}s")

    return pred_x, cfl_ok


def trajectory_loss(pred_x, gt_x):
    """Mean squared particle-position error, skipping frame 0 (known artifact per CLAUDE.md)."""
    n = min(pred_x.shape[0], gt_x.shape[0])
    pred = pred_x[1:n]
    gt = gt_x[1:n]
    return torch.mean((pred - gt) ** 2).item()


def summarize_trajectory(x, n_substeps, num_frames):
    """Per-frame centroid (mean particle position) and bounding-box extent, sampled at frame
    boundaries from a (T, P, 3) substep-indexed position array -- a lightweight summary
    (2 * num_frames * 3 floats) for logging/plotting, not the full per-substep particle tensor.
    Includes frame 0 (a real recorded state for GtX/pred_x, unlike GtF's frame 0 -- see the
    module docstring's dt/frame gotcha reference).
    """
    x = np.asarray(x)
    centroids, extents = [], []
    for f in range(num_frames):
        row = min(f * n_substeps, x.shape[0] - 1)
        frame_x = x[row]
        centroids.append(frame_x.mean(axis=0).tolist())
        extents.append((frame_x.max(axis=0) - frame_x.min(axis=0)).tolist())
    return centroids, extents


def make_objective(cfg, gt_x, gt_x0, gt_v0, num_frames_opt, n_substeps, log, trajectory_summaries,
                    verbose=False):
    def objective(params):
        iteration = len(log)  # shared counter so trajectory_summaries indices line up with
                               # optimizer_iterations_log even if a CFL-violated eval is skipped
        log_E, nu_raw = params
        E = float(np.exp(log_E))
        # squash nu into a physically valid range (avoid the lam singularity at nu=0.5)
        nu = 0.01 + 0.48 / (1.0 + np.exp(-nu_raw))
        mu, lam = lame_from_E_nu(E, nu)

        pred_x, cfl_ok = run_forward(cfg, mu, lam, num_frames_opt, gt_x0=gt_x0, gt_v0=gt_v0,
                                      verbose_timing=verbose)
        if not cfl_ok:
            log.append({"E": E, "nu": nu, "loss": None, "note": "CFL violated"})
            return 1e6  # penalize unstable configs instead of crashing the optimizer

        loss = trajectory_loss(pred_x, gt_x[:pred_x.shape[0]])
        log.append({"E": E, "nu": nu, "loss": loss})

        centroid, extent = summarize_trajectory(pred_x.numpy(), n_substeps, num_frames_opt)
        trajectory_summaries.append({
            "iteration": iteration, "E": E, "nu": nu, "loss": loss,
            "centroid": centroid, "extent": extent,
        })

        print(f"  E={E:9.2f}  nu={nu:.4f}  loss={loss:.6e}")
        return loss

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", required=True,
                         help="e.g. ../output/001a_elastic_sphere_tissue_free_fall_test_soft")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--init_E", type=float, default=8000.0,
                         help="initial guess, Pa (soft-tissue range is 5000-20000 per CLAUDE.md)")
    parser.add_argument("--init_nu", type=float, default=0.35)
    parser.add_argument("--num_frames_opt", type=int, default=None,
                         help="rollout length used during the search. Defaults to the sim's own "
                              "full num_frames: per-call cost is dominated by ti.reset()/ti.init() "
                              "overhead, not substep count (see module docstring's timing note), "
                              "so a short rollout buys little speed while losing the ability to "
                              "identify parameters (like nu) that only show up after floor "
                              "contact. Override to something smaller only if you've confirmed "
                              "that tradeoff doesn't hold for your sim/machine.")
    parser.add_argument("--max_iters", type=int, default=15)
    parser.add_argument("--verbose", action="store_true",
                         help="print per-call setup/physics timing breakdown in addition to the "
                              "E/nu/loss line -- noisy, diagnostic only, off by default")
    args = parser.parse_args()

    cfg, gt_x, gt_v, n_substeps = load_ground_truth(args.sim_dir)
    full_num_frames = cfg["visualization_cfg"]["num_frames"]
    if args.num_frames_opt is None:
        args.num_frames_opt = full_num_frames

    gt_x0 = gt_x[0].numpy().astype(np.float32)
    gt_v0 = gt_v[0].numpy().astype(np.float32)

    true_mu = cfg["objects"]["sphere"]["material"]["mu"]
    true_lam = cfg["objects"]["sphere"]["material"]["lam"]
    true_E, true_nu = E_nu_from_lame(true_mu, true_lam)
    print(f"Ground truth (from config.yaml, for sanity-check comparison only -- "
          f"a real RGB-video inverse problem won't have this): E={true_E:.2f}, nu={true_nu:.4f}")

    log = []
    trajectory_summaries = []
    objective = make_objective(cfg, gt_x, gt_x0, gt_v0, args.num_frames_opt, n_substeps, log,
                                trajectory_summaries, verbose=args.verbose)

    x0 = np.array([np.log(args.init_E), 0.0])  # nu_raw=0 -> nu=0.01+0.24=0.25 at init
    # scipy's default initial-simplex construction perturbs each x0 coordinate that's != 0 by a
    # 5% relative step, but any coordinate that's exactly 0 (nu_raw here) instead gets a tiny
    # fixed absolute step of 0.00025 -- far too small to produce a loss difference distinguishable
    # from noise, so Nelder-Mead had no signal to ever expand the simplex along that axis and nu
    # stayed pinned at its starting value regardless of rollout length. Build the simplex
    # explicitly instead, with a deliberate, comparable step in both dimensions.
    step_E, step_nu = 0.3, 1.0
    initial_simplex = np.array([
        x0,
        x0 + np.array([step_E, 0.0]),
        x0 + np.array([0.0, step_nu]),
    ])
    print(f"\nOptimizing over {args.num_frames_opt}-frame rollouts (max {args.max_iters} iters)...")
    result = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": args.max_iters, "xatol": 1e-2, "fatol": 1e-8,
                                "initial_simplex": initial_simplex})

    recovered_E = float(np.exp(result.x[0]))
    recovered_nu = float(0.01 + 0.48 / (1.0 + np.exp(-result.x[1])))
    recovered_mu, recovered_lam = lame_from_E_nu(recovered_E, recovered_nu)

    print(f"\nRecovered (short-rollout fit): E={recovered_E:.2f}, nu={recovered_nu:.4f}")
    print(f"Ground truth:                  E={true_E:.2f}, nu={true_nu:.4f}")

    # Final validation at the sim's actual full length -- the number that actually matters
    print("\nValidating recovered parameters at full trajectory length...")
    pred_x_full, cfl_ok = run_forward(cfg, recovered_mu, recovered_lam, full_num_frames,
                                       gt_x0=gt_x0, gt_v0=gt_v0, verbose_timing=args.verbose)
    final_loss = trajectory_loss(pred_x_full, gt_x) if cfl_ok else None

    # Ground-truth per-frame centroid/extent, computed once, so the viewer has both curves
    # without needing to load GtX.pt directly in-browser.
    gt_centroid, gt_extent = summarize_trajectory(gt_x.numpy(), n_substeps, full_num_frames)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "sim_dir": args.sim_dir,
            "ground_truth": {"E": true_E, "nu": true_nu, "mu": true_mu, "lam": true_lam},
            "recovered": {"E": recovered_E, "nu": recovered_nu, "mu": recovered_mu, "lam": recovered_lam},
            "short_rollout_loss": float(result.fun),
            "full_rollout_validation_loss": final_loss,
            "num_frames_opt": args.num_frames_opt,
            "optimizer_iterations_log": log,
            "trajectory_summaries": trajectory_summaries,
            "ground_truth_trajectory": {
                "num_frames": full_num_frames,
                "centroid": gt_centroid,
                "extent": gt_extent,
            },
        }, f, indent=2)

    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
