"""
Sim 011: rigid probe poking an elastic tissue sphere.

Time-boxed demo (target: working Blender GIF by 1pm 2026-07-15) -- minimal scope
by design: reuses the 001a (soft elastic, E=15kPa) material config verbatim
(most CFL headroom of the validated elastic presets, ~12x margin, so it's the
safest choice for absorbing whatever extra numerical stress a brand-new,
never-before-run collider interaction adds -- important given the time budget
doesn't allow for an instability-debugging loop). Reuses MPMWrapper completely
unmodified (same construction pattern already proven in
experiments/proxy_net/proxy_net.py's rollout baseline) for particle
initialization, material setup, and the floor collider (which MPMWrapper's
__init__ already picks surface_separate for on material_type=elasticity, per
the earlier per-material floor-collider fix -- nothing new needed there).

The only new physics is a second analytic collider (see
add_sphere_collider() in mpm_simulator_perparticleparams.py) representing the
probe, driven by a hardcoded Python position-over-time function keyed on
substep index -- no config plumbing, no second material, one-way coupling only
(the probe imposes a boundary condition on nearby grid nodes; it is not itself
pushed by the tissue).

Run: conda run -n uniphy python dataset_generate_011_probe_poke.py
"""

import os
import time

import numpy as np
import taichi as ti
import torch
import yaml

from mpm_simulator_perparticleparams import MPMSimulator
from mpmwrapper_perparticleparams import MPMWrapper

REPO_ROOT = "/home/elec594/Desktop/surgical-robotics-tissue-sim"
SAVE_DIR = f"{REPO_ROOT}/simulation/output/011_elastic_sphere_probe_poke_test"

# ── Material: 001a (soft elastic, E~15kPa) verbatim ─────────────────────────
MATERIAL_CFG = dict(
    material_type="MPMSimulator.elasticity",
    object_material="elastic",
    rho=928.8232,
    velocity=[0, 0, 0],
    mu=5172,
    lam=46552,
    yield_stress=1.0,
    plastic_viscosity=1.0,
    friction_alpha=0.0,
    inv_dt=4800,
    inv_frame_dt=24.0,
)
GEOMETRY_CFG = dict(
    sdf_func="sphere",
    sdf_params=[0.1],
    num_particles=500,
    rot_x=45, rot_y=90, rot_z=45,
    pos_x=0.55, pos_y=0.55, pos_z=0,
)
SIMULATOR_CFG = dict(
    dtype="float32",
    BC={"ground": [[0, 0, 0], [0, 1, 0]]},
    gravity=[0, -9.8, 0],
)

NUM_FRAMES = 10               # matches every other sim's convention
PARTICLES_TI_ROOT = 1024
CUDA_CHUNK_SIZE = 2048        # 2000 total substeps fits within this -- no need
                               # to reintroduce the old fixed-buffer bug or size
                               # up, unlike the stiff-elastic (001b) case

# ── Probe: hardcoded position-over-time trajectory, keyed on substep index ──
# Sim-space axes: X,Z horizontal, Y up (matches the floor collider's [0,1,0]
# normal and the 01_animate_sphere.py coordinate remap). The tissue sphere
# starts at (0.55, 0.55, 0), free-falls under existing (unmodified) gravity +
# floor physics and settles by roughly substep ~1400 (well past the ~1425-sub
# step floor-contact time already established for this material in the
# 001a/001b validation notes) -- the probe stays parked well out of the way
# until then, so the free-fall + initial floor impact is exactly the same
# physics as the validated 001a run, undisturbed.
PROBE_RADIUS = 0.06
PROBE_X, PROBE_Z = 0.55, 0.0
PROBE_Y_PARK = 1.0     # high above, out of the way during free-fall/settle
PROBE_Y_POKE = 0.02    # deep enough to guarantee contact with the settled
                        # sphere regardless of exact settle height (this is a
                        # "definitely make contact" choice, not a tuned
                        # percentage-indentation target -- Step 2 only asks to
                        # confirm visible deformation happens, not hit an
                        # exact depth)
SUBSTEP_PARK_END = 1400     # frames 0-7: park, undisturbed free-fall + settle
SUBSTEP_DESCEND_END = 1700  # frames 7-8.5: descend
SUBSTEP_HOLD_END = 1850     # hold (poke)
SUBSTEP_TOTAL = 2000         # frames 9.25-10: withdraw back to park


def probe_y(substep_i):
    if substep_i < SUBSTEP_PARK_END:
        return PROBE_Y_PARK
    if substep_i < SUBSTEP_DESCEND_END:
        t = (substep_i - SUBSTEP_PARK_END) / (SUBSTEP_DESCEND_END - SUBSTEP_PARK_END)
        return PROBE_Y_PARK + t * (PROBE_Y_POKE - PROBE_Y_PARK)
    if substep_i < SUBSTEP_HOLD_END:
        return PROBE_Y_POKE
    t = (substep_i - SUBSTEP_HOLD_END) / (SUBSTEP_TOTAL - SUBSTEP_HOLD_END)
    t = min(t, 1.0)
    return PROBE_Y_POKE + t * (PROBE_Y_PARK - PROBE_Y_POKE)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda"

    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=0.5)

    objects_cfg = {"sphere": {"geometry": GEOMETRY_CFG, "material": MATERIAL_CFG}}
    wrapper = MPMWrapper(objects_cfg, SIMULATOR_CFG, particles_ti_root=PARTICLES_TI_ROOT,
                          cuda_chunk_size=CUDA_CHUNK_SIZE)
    wrapper.initialize_particles()
    wrapper.simulator_variables_initialize()

    # Second analytic collider: the probe. surface_separate, matching elastic's
    # floor treatment -- the tissue can be pushed away and spring back, not
    # glued to the probe.
    probe_center = ti.Vector.field(3, dtype=ti.f32, shape=())
    probe_center[None] = ti.Vector([PROBE_X, PROBE_Y_PARK, PROBE_Z])
    wrapper.simulator.add_sphere_collider(probe_center, PROBE_RADIUS,
                                           surface=MPMSimulator.surface_separate)

    substep_gt = wrapper.simulator.n_substeps[None]
    total_substeps = NUM_FRAMES * substep_gt
    assert total_substeps == SUBSTEP_TOTAL, \
        f"probe trajectory breakpoints assume {SUBSTEP_TOTAL} substeps, got {total_substeps}"

    buffer_len = total_substeps + 2
    particle_x = torch.zeros([PARTICLES_TI_ROOT, buffer_len, 3], dtype=torch.float32).to(device)
    particle_v = torch.zeros([PARTICLES_TI_ROOT, buffer_len, 3], dtype=torch.float32).to(device)
    particle_C = torch.zeros([PARTICLES_TI_ROOT, buffer_len, 3, 3], dtype=torch.float32).to(device)
    particle_F = torch.zeros([PARTICLES_TI_ROOT, buffer_len, 3, 3], dtype=torch.float32).to(device)
    particle_Ftmp = torch.zeros([PARTICLES_TI_ROOT, buffer_len, 3, 3], dtype=torch.float32).to(device)
    particle_stress = torch.zeros([PARTICLES_TI_ROOT, buffer_len, 3, 3], dtype=torch.float32).to(device)

    probe_y_trace = np.zeros(total_substeps, dtype=np.float32)

    print(f"Running {total_substeps} substeps ({NUM_FRAMES} frames x {substep_gt} substeps/frame)...")
    t0 = time.perf_counter()
    cfl_flag = False
    for i in range(total_substeps):
        y = probe_y(i)
        probe_y_trace[i] = y
        probe_center[None] = ti.Vector([PROBE_X, y, PROBE_Z])

        if wrapper.simulator.cfl_satisfy[None]:
            wrapper.simulator.substep(i)
        if not wrapper.simulator.cfl_satisfy[None]:
            print(f"CFL not satisfied at substep {i}")
            cfl_flag = True

        wrapper.simulator.get_x_gt(i, particle_x)
        wrapper.simulator.get_v_gt(i, particle_v)
        wrapper.simulator.get_C(i, particle_C)
        wrapper.simulator.get_F(i + 1, particle_F)
        wrapper.simulator.get_Ftmp(i + 1, particle_Ftmp)
        wrapper.simulator.get_stress(i + 1, particle_stress)

    ti.sync()
    wall_time = time.perf_counter() - t0
    print(f"Done in {wall_time:.1f}s. CFL violated at any point: {cfl_flag}")

    n = wrapper.num_particles[None]
    final_x = particle_x[:n, :buffer_len].permute(1, 0, 2)
    final_v = particle_v[:n, :buffer_len].permute(1, 0, 2)
    final_C = particle_C[:n, :buffer_len].permute(1, 0, 2, 3)
    final_F = particle_F[:n, :buffer_len].permute(1, 0, 2, 3)
    final_Ftmp = particle_Ftmp[:n, :buffer_len].permute(1, 0, 2, 3)
    final_stress = particle_stress[:n, :buffer_len].permute(1, 0, 2, 3)

    torch.save(final_x, f"{SAVE_DIR}/GtX.pt")
    torch.save(final_v, f"{SAVE_DIR}/GtV.pt")
    torch.save(final_C, f"{SAVE_DIR}/GtC.pt")
    torch.save(final_F, f"{SAVE_DIR}/GtF.pt")
    torch.save(final_Ftmp, f"{SAVE_DIR}/GtFtmp.pt")
    torch.save(final_stress, f"{SAVE_DIR}/GtStress.pt")

    full_cfg = dict(
        objects=dict(sphere=dict(geometry=GEOMETRY_CFG, material=MATERIAL_CFG)),
        simulator_cfg=SIMULATOR_CFG,
        visualization_cfg=dict(num_frames=NUM_FRAMES),
        train_cfg=dict(save_dir="011_elastic_sphere_probe_poke_test", local_dir=f"{REPO_ROOT}/simulation/output",
                        particles_ti_root=PARTICLES_TI_ROOT, cuda_chunk_size=CUDA_CHUNK_SIZE),
        probe=dict(
            radius=PROBE_RADIUS, x=PROBE_X, z=PROBE_Z, y_park=PROBE_Y_PARK, y_poke=PROBE_Y_POKE,
            substep_park_end=SUBSTEP_PARK_END, substep_descend_end=SUBSTEP_DESCEND_END,
            substep_hold_end=SUBSTEP_HOLD_END, substep_total=SUBSTEP_TOTAL,
            note="One-way analytic sphere collider (surface_separate), NOT a real rigid body -- "
                 "see add_sphere_collider() in mpm_simulator_perparticleparams.py.",
        ),
    )
    with open(f"{SAVE_DIR}/config.yaml", "w") as f:
        yaml.dump(full_cfg, f)

    # 40-sample downsampled probe center trajectory, same NUM_SAMPLES convention
    # as sphere_keyframes.npy, for the Blender render step to consume directly
    # (single source of truth for the trajectory -- Blender doesn't re-derive it).
    NUM_SAMPLES = 40
    sample_idx = np.linspace(0, total_substeps - 1, NUM_SAMPLES).round().astype(int)
    probe_keyframes = np.zeros((NUM_SAMPLES, 3), dtype=np.float32)
    for row, si in enumerate(sample_idx):
        probe_keyframes[row] = [PROBE_X, probe_y_trace[si], PROBE_Z]
    np.save(f"{SAVE_DIR}/probe_keyframes.npy", probe_keyframes)

    print(f"Saved to {SAVE_DIR}")
    return cfl_flag


if __name__ == "__main__":
    cfl_flag = main()
    if cfl_flag:
        raise SystemExit("CFL violated at some point during the run -- see output above")
