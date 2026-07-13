"""
Proxy network experiment (Prof. Boominathan's suggestion) -- ILLUSTRATIVE ONLY, N=2.

Question being tested: does a cheap, per-trajectory descriptor + initial state carry
enough information for a *direct* (initial state, descriptor) -> final state map to
work at all, skipping the timestep-by-timestep MPM rollout entirely? This is a sanity
check, not a claim that the resulting model generalizes or should replace the real
pipeline.

READ THIS BEFORE INTERPRETING ANY NUMBER BELOW:

1. There is no learned latent z / g_phi / f_theta anywhere in this repo. That
   architecture (a learned material latent decoded by two small networks inside the
   MPM stress update) belongs to `simulation/nclaw/`, which was deleted in an earlier
   session -- confirmed via repo-wide grep (no "g_phi"/"f_theta"/"latent" hits in any
   code path) and confirmed no pretrained weights are vendored in this repo or
   downloaded anywhere else on this machine (no .pth/.ckpt files, no ckpt/ folder).
   The current pipeline (mpm_simulator_perparticleparams.py) is a pure forward MPM
   solver driven by explicit scalar material parameters, not a learned latent.

   This script therefore uses those 5 explicit parameters (mu, lam, yield_stress,
   plastic_viscosity, friction_alpha), read straight from each sim's config.yaml, as
   a STAND-IN conditioning vector -- NOT a learned latent. A low error below shows
   only that a network can map (initial state, material params) -> final state for
   these two specific trajectories, not that a learned z would carry equivalent
   information, and not that this generalizes to unseen materials.

2. With exactly 2 trajectories, the proxy MLP is trained AND evaluated on the same 2
   examples -- there is no train/test split (nothing meaningful to hold out from 2
   points). A very small reconstruction error here mostly demonstrates that a
   sufficiently expressive network can memorize a discriminating-key mapping
   (descriptor A -> output A, descriptor B -> output B). It does NOT demonstrate
   generalization to a third material. This is stated here explicitly per the
   experiment's own design brief.

3. "Rollout baseline": there is no separately-trained inference network to run, so
   the "baseline" is the actual existing forward MPM solver
   (mpm_simulator_perparticleparams.py / mpmwrapper_perparticleparams.py, imported
   here completely unmodified), re-run from each sim's real recorded starting state
   (GtX[0]/GtV[0], not a fresh random SDF sample) for exactly 1999 substeps -- matching
   how index 1999 (the last row the original save loop ever wrote; indices 2000/2001
   are unwritten zero buffer per the repo's known trim convention) was originally
   produced. Because this is a deterministic re-run of the same physics with the same
   inputs, its reconstruction error against GtX[1999] should be ~0 by construction --
   that's a correctness check on the harness, not a finding. The real rollout-vs-proxy
   comparison that matters here is WALL-CLOCK TIME, not accuracy.

Materials used (2 trajectories, as specified): elastic sim 001a (soft/jelly) and
plasticine sim 002a (runny putty). The old undifferentiated "001"/"002" sims no
longer exist on disk (superseded by paired extreme-parameter variants in an earlier
session) -- 001a/002a were picked as the "a" variant of each material; swap the
SIMS dict below to use 001b/002b or any other pair instead.
"""

import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

REPO_ROOT = "/home/elec594/Desktop/surgical-robotics-tissue-sim"
DATA_PROCESS_DIR = f"{REPO_ROOT}/simulation/data_process"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

sys.path.insert(0, DATA_PROCESS_DIR)

SIMS = {
    "elastic (001a, soft)": {
        "dir": f"{REPO_ROOT}/simulation/output/001a_elastic_sphere_tissue_free_fall_test_soft",
        "short": "elastic_001a",
    },
    "plasticine (002a, runny)": {
        "dir": f"{REPO_ROOT}/simulation/output/002a_plasticine_sphere_free_fall_test_runny",
        "short": "plasticine_002a",
    },
}

DESCRIPTOR_KEYS = ["mu", "lam", "yield_stress", "plastic_viscosity", "friction_alpha"]
N_ROLLOUT_SUBSTEPS = 1999  # reproduces GtX[1999], the last row the save loop ever wrote
PARTICLES_TI_ROOT = 1024
CUDA_CHUNK_SIZE = 2048


def load_material_cfg(sim_dir):
    with open(f"{sim_dir}/config.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["objects"]["sphere"]


def run_rollout_baseline(sim_dir):
    """Re-runs the actual existing forward MPM solver (unmodified import) from this
    sim's real recorded starting state, timing the full 1999-substep rollout.
    Returns (final_positions_computed[500,3], wall_clock_seconds)."""
    import taichi as ti
    from mpmwrapper_perparticleparams import MPMWrapper

    x0 = torch.load(f"{sim_dir}/GtX.pt", map_location="cpu")[0].numpy().astype(np.float32)
    v0 = torch.load(f"{sim_dir}/GtV.pt", map_location="cpu")[0].numpy().astype(np.float32)

    sphere_cfg = load_material_cfg(sim_dir)
    with open(f"{sim_dir}/config.yaml") as f:
        full_cfg = yaml.safe_load(f)
    simulator_cfg = full_cfg["simulator_cfg"]
    objects_cfg = {"sphere": {"geometry": sphere_cfg["geometry"], "material": sphere_cfg["material"]}}

    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=0.5)

    wrapper = MPMWrapper(objects_cfg, simulator_cfg, particles_ti_root=PARTICLES_TI_ROOT,
                          cuda_chunk_size=CUDA_CHUNK_SIZE)
    wrapper.initialize_particles()
    # Override the freshly-(randomly-)sampled SDF particles with this sim's *actual*
    # recorded starting state, so the rollout reproduces the real trajectory exactly
    # rather than a fresh random one.
    wrapper.init_particles = x0
    wrapper.init_velocities = v0
    wrapper.simulator_variables_initialize()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ti.sync()
    t0 = time.perf_counter()
    for i in range(N_ROLLOUT_SUBSTEPS):
        if wrapper.simulator.cfl_satisfy[None]:
            wrapper.simulator.substep(i)
    ti.sync()
    wall_time = time.perf_counter() - t0

    final_x = wrapper.simulator.x.to_numpy()[:500, N_ROLLOUT_SUBSTEPS, :]
    cfl_ok = bool(wrapper.simulator.cfl_satisfy[None])
    return final_x.copy(), wall_time, cfl_ok


class ProxyMLP(nn.Module):
    """Deliberately tiny -- a couple hidden layers, per the experiment brief. Input is
    (initial positions [1500] ++ initial velocities [1500] ++ 5-dim material
    descriptor stand-in), output is final positions [1500]."""

    def __init__(self, n_particles=500, descriptor_dim=5):
        super().__init__()
        in_dim = n_particles * 3 * 2 + descriptor_dim
        out_dim = n_particles * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    results = {}

    print("=" * 78)
    print("PHASE 1: rollout baseline (actual forward MPM solver, unmodified import)")
    print("=" * 78)
    for label, info in SIMS.items():
        sim_dir = info["dir"]
        print(f"\n--- {label} ---")
        final_x_computed, wall_time, cfl_ok = run_rollout_baseline(sim_dir)

        gt_x = torch.load(f"{sim_dir}/GtX.pt", map_location="cpu").numpy()
        ground_truth_final = gt_x[N_ROLLOUT_SUBSTEPS]  # index 1999, last row ever written

        rollout_err = float(np.linalg.norm(final_x_computed - ground_truth_final, axis=1).mean())
        print(f"  CFL satisfied throughout: {cfl_ok}")
        print(f"  rollout wall time ({N_ROLLOUT_SUBSTEPS} substeps): {wall_time:.4f} s")
        print(f"  rollout mean per-particle L2 error vs GtX[1999]: {rollout_err:.6e}")
        print("  (expected ~0: this is a deterministic re-run of the same physics from")
        print("   the same recorded starting state, so this number is a correctness")
        print("   check on the harness, not a finding.)")

        results[label] = dict(
            sim_dir=sim_dir,
            short=info["short"],
            initial_positions=gt_x[0].copy(),
            initial_velocities=torch.load(f"{sim_dir}/GtV.pt", map_location="cpu").numpy()[0].copy(),
            ground_truth_final=ground_truth_final.copy(),
            rollout_final=final_x_computed,
            rollout_time_s=wall_time,
            rollout_error=rollout_err,
        )

    print("\n" + "=" * 78)
    print("PHASE 2: proxy network (illustrative only, N=2 -- see module docstring)")
    print("=" * 78)

    labels = list(SIMS.keys())
    descriptors = []
    initial_states = []
    targets = []
    for label in labels:
        sim_dir = SIMS[label]["dir"]
        mat_cfg = load_material_cfg(sim_dir)["material"]
        descriptor = np.array([mat_cfg[k] for k in DESCRIPTOR_KEYS], dtype=np.float32)
        descriptors.append(descriptor)
        r = results[label]
        flat_input = np.concatenate([
            r["initial_positions"].flatten(),
            r["initial_velocities"].flatten(),
            descriptor,
        ])
        initial_states.append(flat_input)
        targets.append(r["ground_truth_final"].flatten())

    print(f"\nStand-in descriptor (mu, lam, yield_stress, plastic_viscosity, friction_alpha):")
    for label, d in zip(labels, descriptors):
        print(f"  {label}: {d.tolist()}")

    X = torch.tensor(np.stack(initial_states), dtype=torch.float32)
    Y = torch.tensor(np.stack(targets), dtype=torch.float32)

    # Normalize inputs/outputs for stable training -- purely a numerics convenience,
    # doesn't change what's being tested.
    x_mean, x_std = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    y_mean, y_std = Y.mean(0, keepdim=True), Y.std(0, keepdim=True).clamp_min(1e-6)
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    torch.manual_seed(0)
    model = ProxyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("\nTraining proxy MLP on N=2 examples (memorization, not generalization)...")
    n_epochs = 3000
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(Xn)
        loss = loss_fn(pred, Yn)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:5d}  train MSE (normalized) = {loss.item():.6e}")

    # Time a single forward pass (averaged over many reps for a stable estimate).
    model.eval()
    n_reps = 1000
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = model(Xn[:1])
        proxy_time_per_call = (time.perf_counter() - t0) / n_reps

        proxy_pred_n = model(Xn)
        proxy_pred = (proxy_pred_n * y_std + y_mean).numpy().reshape(len(labels), 500, 3)

    print(f"\nProxy forward pass wall time (mean over {n_reps} calls, batch size 1): "
          f"{proxy_time_per_call:.6e} s")

    for i, label in enumerate(labels):
        gt = results[label]["ground_truth_final"]
        proxy_err = float(np.linalg.norm(proxy_pred[i] - gt, axis=1).mean())
        results[label]["proxy_final"] = proxy_pred[i]
        results[label]["proxy_error"] = proxy_err
        results[label]["proxy_time_s"] = proxy_time_per_call
        print(f"  {label}: proxy mean per-particle L2 error vs GtX[1999] = {proxy_err:.6e}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'Material':<26} {'Rollout err':>13} {'Proxy err':>13} {'Rollout time (s)':>18} {'Proxy time (s)':>16}"
    print(header)
    print("-" * len(header))
    for label in labels:
        r = results[label]
        print(f"{label:<26} {r['rollout_error']:>13.4e} {r['proxy_error']:>13.4e} "
              f"{r['rollout_time_s']:>18.4f} {r['proxy_time_s']:>16.4e}")
    speedup = np.mean([results[l]["rollout_time_s"] / results[l]["proxy_time_s"] for l in labels])
    print(f"\nProxy forward pass is ~{speedup:,.0f}x faster than the {N_ROLLOUT_SUBSTEPS}-substep "
          f"rollout (expected -- one MLP forward pass vs {N_ROLLOUT_SUBSTEPS} physics substeps).")
    print("Reminder: proxy error is a TRAINING-set reconstruction error over N=2 examples,")
    print("not a held-out generalization error -- see module docstring caveat #2.")

    # --- plot ---
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 6))
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        r = results[label]
        gt, roll, prox = r["ground_truth_final"], r["rollout_final"], r["proxy_final"]
        ax.scatter(gt[:, 0], gt[:, 1], s=14, c="#444444", alpha=0.6, label="ground truth (GtX[1999])")
        ax.scatter(roll[:, 0], roll[:, 1], s=10, c="#1f77b4", alpha=0.6, marker="x", label="rollout (re-run solver)")
        ax.scatter(prox[:, 0], prox[:, 1], s=10, c="#ff7f0e", alpha=0.6, marker="+", label="proxy MLP")
        ax.set_title(f"{label}\nrollout err={r['rollout_error']:.2e}  proxy err={r['proxy_error']:.2e}")
        ax.set_xlabel("x (sim units)")
        ax.set_ylabel("y (sim units, vertical)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Proxy network experiment -- ILLUSTRATIVE ONLY, N=2 trajectories "
                 "(stand-in 5-dim material descriptor, not a learned latent z)", fontsize=10)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "comparison.png")
    fig.savefig(plot_path, dpi=130)
    print(f"\nSaved comparison plot to {plot_path}")

    # --- json summary ---
    summary = {
        label: {
            "rollout_error": results[label]["rollout_error"],
            "proxy_error": results[label]["proxy_error"],
            "rollout_time_s": results[label]["rollout_time_s"],
            "proxy_time_s": results[label]["proxy_time_s"],
        }
        for label in labels
    }
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
