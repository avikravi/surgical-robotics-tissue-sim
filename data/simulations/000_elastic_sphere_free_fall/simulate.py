"""
000_elastic_sphere_free_fall -- UniPhy-based MLS-MPM physics validation
=========================================================================

A solid elastic sphere falls under gravity and lands on a floor boundary.
This is a from-scratch, standalone re-implementation of the MLS-MPM solver
core used in UniPhy (CVPR 2025) / NCLaw (nclaw/sim/mpm.py), using the
standard `warp-lang` pip package -- no custom Warp build required.

Reference repo : https://github.com/HimangiM/UniPhy_CVPR2025
Reference paper: Mittal et al., "UniPhy: Learning a Unified Constitutive
                  Model for Inverse Physics Simulation", CVPR 2025.

Parameter provenance
---------------------
- Simulation constants (num_grids, dt, bound, gravity) match UniPhy's
  "low" quality preset: experiments/configs/sim/low.yaml
- Sphere geometry (radius=0.1) matches UniPhy's sphere object preset:
  configs/objects/sphere/geometry/default.yaml
- Material (mu=350, lam=500, rho=928.8232) is the LEAST-STIFF boundary of
  UniPhy's elastic dataset range (data_process/elastic.py randomizes
  mu in [350, 2595196] and lam in [500, 2580120], rho fixed per
  configs/objects/box/material/elastic.yaml)

This script is intentionally minimal: one elastic object, free fall,
floor collision. The goal is to validate the core physics (gravity
integration, floor boundary, corotated elasticity) before building up
to multi-object / multi-material tissue scenes.

Run in Google Colab with a GPU runtime (T4 or better):
    !pip install warp-lang -q
    python 000_elastic_sphere_free_fall_uniphy_mpm.py
"""

import os
import shutil
import json

import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================================
# Configuration
# =========================================================================

# ---- Simulator constants (UniPhy "low" quality: experiments/configs/sim/low.yaml) ----
NUM_GRIDS = 20                 # background grid is NUM_GRIDS^3 cells
DX = 1.0 / NUM_GRIDS           # grid cell size
DT = 5e-4                      # timestep (s)
GRAVITY = np.array([0.0, -9.8, 0.0], dtype=np.float32)  # m/s^2, -y is down
BOUND = 3                      # grid cells reserved as floor/wall boundary
EPS = 1e-7                     # avoids divide-by-zero in empty grid cells
NUM_STEPS = 4000               # 4000 * 5e-4 = 2.0 s -- long enough to see if the bounce/jiggle settles
SAVE_EVERY = 10                # record a frame every 10 steps -> 400 frames (13.3s @ 30fps)

# ---- Object geometry: sphere (UniPhy sphere preset radius = 0.1) ----
SPHERE_CENTER = np.array([0.5, 0.6, 0.5])
SPHERE_RADIUS = 0.1            # configs/objects/sphere/geometry/default.yaml: sdf_params: [0.1]
RES = 16                        # particles-per-side of the bounding cube before carving

# ---- Material: UniPhy "jelly" preset ----
# experiments/configs/env/blob/material/elasticity/corotated_jelly.yaml (E=1e5, nu=0.2)
# experiments/configs/env/blob/jelly.yaml (rho=1e3)
RHO = 1000.0                     # kg/m^3
E = 1e5                          # Young's modulus (Pa)
NU = 0.2                         # Poisson's ratio
MU = E / (2.0 * (1.0 + NU))                                  # Lame's first parameter
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))               # Lame's second parameter

OUTPUT_DIR = "uniphy_mpm_output"
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUTPUT_DIR, "elastic_sphere_free_fall.mp4")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")


# =========================================================================
# Geometry: solid sphere of particles
# =========================================================================

def build_sphere_particles():
    lin = np.linspace(-SPHERE_RADIUS, SPHERE_RADIUS, RES)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing='ij'), axis=-1).reshape(-1, 3)

    dist_from_center = np.linalg.norm(grid, axis=1)
    sphere_points = grid[dist_from_center <= SPHERE_RADIUS]

    positions = (sphere_points + SPHERE_CENTER).astype(np.float32)
    fill_fraction = sphere_points.shape[0] / grid.shape[0]

    print(f"Bounding-cube lattice points: {grid.shape[0]}")
    print(f"Particles inside sphere:      {positions.shape[0]}")
    print(f"Fill fraction: {fill_fraction:.3f} (sphere ~0.52, cube would be 1.0)")

    return positions


# =========================================================================
# Warp kernels: MLS-MPM core (P2G -> grid update -> G2P) + corotated elasticity
# =========================================================================

@wp.kernel
def clear_grid(
    grid_m: wp.array(dtype=float, ndim=3),
    grid_mv: wp.array(dtype=wp.vec3, ndim=3),
    grid_v: wp.array(dtype=wp.vec3, ndim=3),
):
    i, j, k = wp.tid()
    grid_m[i, j, k] = 0.0
    grid_mv[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
    grid_v[i, j, k] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def compute_stress_corotated(
    F: wp.array(dtype=wp.mat33),
    mu: float,
    lam: float,
    stress: wp.array(dtype=wp.mat33),
):
    p = wp.tid()
    Fp = F[p]

    # F = U * diag(sigma) * V^T  (SVD)
    U, sigma, V = wp.svd3(Fp)
    R = U * wp.transpose(V)          # closest pure rotation to F
    J = sigma[0] * sigma[1] * sigma[2]  # volume ratio

    I33 = wp.matrix_from_cols(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0))

    shear_term = (2.0 * mu) * ((Fp - R) * wp.transpose(Fp))
    volume_term = (lam * J * (J - 1.0)) * I33

    stress[p] = shear_term + volume_term


@wp.kernel
def p2g(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    stress: wp.array(dtype=wp.mat33),
    p_mass: float,
    p_vol: float,
    dx: float,
    inv_dx: float,
    dt: float,
    grid_m: wp.array(dtype=float, ndim=3),
    grid_mv: wp.array(dtype=wp.vec3, ndim=3),
):
    p = wp.tid()

    pos = x[p] * inv_dx
    base_x = int(pos[0] - 0.5)
    base_y = int(pos[1] - 0.5)
    base_z = int(pos[2] - 0.5)
    fx = pos - wp.vec3(float(base_x), float(base_y), float(base_z))

    w0 = 0.5 * wp.cw_mul(wp.vec3(1.5) - fx, wp.vec3(1.5) - fx)
    w1 = wp.vec3(0.75) - wp.cw_mul(fx - wp.vec3(1.0), fx - wp.vec3(1.0))
    w2 = 0.5 * wp.cw_mul(fx - wp.vec3(0.5), fx - wp.vec3(0.5))
    w = wp.matrix_from_cols(w0, w1, w2)

    affine_force = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * stress[p]
    affine = affine_force + p_mass * C[p]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[0, i] * w[1, j] * w[2, k]
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx

                m_contrib = weight * p_mass
                mv_contrib = weight * (p_mass * v[p] + affine * dpos)

                wp.atomic_add(grid_m, base_x + i, base_y + j, base_z + k, m_contrib)
                wp.atomic_add(grid_mv, base_x + i, base_y + j, base_z + k, mv_contrib)


@wp.kernel
def grid_op(
    grid_m: wp.array(dtype=float, ndim=3),
    grid_mv: wp.array(dtype=wp.vec3, ndim=3),
    grid_v: wp.array(dtype=wp.vec3, ndim=3),
    gravity: wp.vec3,
    dt: float,
    bound: int,
    num_grids: int,
    eps: float,
):
    px, py, pz = wp.tid()

    m = grid_m[px, py, pz]
    if m > 0.0:
        v = grid_mv[px, py, pz] / (m + eps) + gravity * dt
    else:
        v = gravity * dt

    # floor / walls: "free-slip" boundary -- zero the velocity component
    # pointing into the wall, but allow sliding along it
    if px < bound and v[0] < 0.0:
        v = wp.vec3(0.0, v[1], v[2])
    if py < bound and v[1] < 0.0:
        v = wp.vec3(v[0], 0.0, v[2])
    if pz < bound and v[2] < 0.0:
        v = wp.vec3(v[0], v[1], 0.0)
    if px > num_grids - bound and v[0] > 0.0:
        v = wp.vec3(0.0, v[1], v[2])
    if py > num_grids - bound and v[1] > 0.0:
        v = wp.vec3(v[0], 0.0, v[2])
    if pz > num_grids - bound and v[2] > 0.0:
        v = wp.vec3(v[0], v[1], 0.0)

    grid_v[px, py, pz] = v


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33),
    F: wp.array(dtype=wp.mat33),
    grid_v: wp.array(dtype=wp.vec3, ndim=3),
    dx: float,
    inv_dx: float,
    dt: float,
    clip_bound: float,
):
    p = wp.tid()

    pos = x[p] * inv_dx
    base_x = int(pos[0] - 0.5)
    base_y = int(pos[1] - 0.5)
    base_z = int(pos[2] - 0.5)
    fx = pos - wp.vec3(float(base_x), float(base_y), float(base_z))

    w0 = 0.5 * wp.cw_mul(wp.vec3(1.5) - fx, wp.vec3(1.5) - fx)
    w1 = wp.vec3(0.75) - wp.cw_mul(fx - wp.vec3(1.0), fx - wp.vec3(1.0))
    w2 = 0.5 * wp.cw_mul(fx - wp.vec3(0.5), fx - wp.vec3(0.5))
    w = wp.matrix_from_cols(w0, w1, w2)

    zero3 = wp.vec3(0.0, 0.0, 0.0)
    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.matrix_from_cols(zero3, zero3, zero3)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[0, i] * w[1, j] * w[2, k]
                dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
                grid_vel = grid_v[base_x + i, base_y + j, base_z + k]

                new_v = new_v + weight * grid_vel
                new_C = new_C + (4.0 * weight * inv_dx * inv_dx) * wp.outer(grid_vel, dpos)

    v[p] = new_v
    C[p] = new_C

    I33 = wp.matrix_from_cols(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0))
    F[p] = (I33 + dt * new_C) * F[p]

    new_x = x[p] + dt * new_v
    bound = clip_bound * dx
    new_x = wp.vec3(
        wp.clamp(new_x[0], bound, 1.0 - bound),
        wp.clamp(new_x[1], bound, 1.0 - bound),
        wp.clamp(new_x[2], bound, 1.0 - bound),
    )
    x[p] = new_x


# =========================================================================
# Rendering helpers
# =========================================================================

BASE_COLOR = np.array([0.15, 0.75, 0.55])  # jelly-like teal/green
LIGHT_DIR = np.array([-0.4, 0.6, 0.5])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


def shade_colors(frame):
    center = frame.mean(axis=0)
    normals = frame - center
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms
    brightness = np.clip(normals @ LIGHT_DIR, 0.0, 1.0) * 0.7 + 0.3
    return np.clip(BASE_COLOR[None, :] * brightness[:, None], 0, 1)


def render_video(trajectory, time_history, floor_y):
    shutil.rmtree(FRAME_DIR, ignore_errors=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    xx, zz = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
    yy_floor = np.full_like(xx, floor_y)

    for i, frame in enumerate(trajectory):
        ax.cla()

        colors = shade_colors(frame)
        # NOTE: (x, z, y) order -- y (gravity/up) is drawn as the vertical screen axis
        ax.scatter(frame[:, 0], frame[:, 2], frame[:, 1], s=25, c=colors, edgecolors='none')
        ax.plot_surface(xx, zz, yy_floor, alpha=0.3, color='burlywood')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('y (up)')
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f't = {time_history[i]:.3f} s')

        fig.savefig(f"{FRAME_DIR}/frame_{i:04d}.png", dpi=100)

        if i % 50 == 0:
            print(f"Rendered frame {i}/{len(trajectory)}")

    plt.close(fig)

    sizes = {Image.open(f"{FRAME_DIR}/frame_{i:04d}.png").size for i in range(len(trajectory))}
    print("Unique frame sizes:", sizes)

    os.system(
        f"ffmpeg -y -framerate 30 -i {FRAME_DIR}/frame_%04d.png "
        f"-pix_fmt yuv420p -vcodec libx264 {VIDEO_PATH}"
    )
    print(f"Video written to {VIDEO_PATH}")


# =========================================================================
# Main
# =========================================================================

def main():
    wp.init()
    print("Warp version:", wp.config.version)
    print("Using device:", wp.get_device())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nMaterial (UniPhy jelly preset: E=1e5 Pa, nu=0.2, rho=1000):")
    print(f"  mu={MU:.2f}, lam={LAM:.2f}, rho={RHO}")
    print(f"  equivalent: E={E:.3e} Pa, nu={NU:.4f}")

    positions = build_sphere_particles()
    n_particles = positions.shape[0]
    particle_vol = (2 * SPHERE_RADIUS / RES) ** 3
    particle_mass = particle_vol * RHO
    floor_y = BOUND * DX
    clip_bound = float(BOUND)

    print(f"\nParticle volume: {particle_vol:.3e} m^3")
    print(f"Particle mass:   {particle_mass:.3e} kg")
    print(f"Total mass:      {particle_mass * n_particles:.3e} kg")
    print(f"Floor at y = {floor_y:.3f}")
    print(f"Drop height above floor: {positions[:,1].min() - floor_y:.3f}")

    # ---- state arrays ----
    particle_x = wp.from_numpy(positions, dtype=wp.vec3, device="cuda")
    particle_v = wp.zeros(n_particles, dtype=wp.vec3, device="cuda")
    particle_C = wp.zeros(n_particles, dtype=wp.mat33, device="cuda")
    particle_F = wp.from_numpy(
        np.tile(np.eye(3, dtype=np.float32), (n_particles, 1, 1)), dtype=wp.mat33, device="cuda")
    particle_stress = wp.zeros(n_particles, dtype=wp.mat33, device="cuda")

    grid_m = wp.zeros((NUM_GRIDS, NUM_GRIDS, NUM_GRIDS), dtype=float, device="cuda")
    grid_mv = wp.zeros((NUM_GRIDS, NUM_GRIDS, NUM_GRIDS), dtype=wp.vec3, device="cuda")
    grid_v = wp.zeros((NUM_GRIDS, NUM_GRIDS, NUM_GRIDS), dtype=wp.vec3, device="cuda")

    def simulation_step():
        wp.launch(compute_stress_corotated, dim=n_particles,
                  inputs=[particle_F, MU, LAM, particle_stress])
        wp.launch(clear_grid, dim=[NUM_GRIDS, NUM_GRIDS, NUM_GRIDS],
                  inputs=[grid_m, grid_mv, grid_v])
        wp.launch(p2g, dim=n_particles,
                  inputs=[particle_x, particle_v, particle_C, particle_stress,
                          particle_mass, particle_vol, DX, 1.0 / DX, DT,
                          grid_m, grid_mv])
        wp.launch(grid_op, dim=[NUM_GRIDS, NUM_GRIDS, NUM_GRIDS],
                  inputs=[grid_m, grid_mv, grid_v,
                          wp.vec3(*GRAVITY), DT, BOUND, NUM_GRIDS, EPS])
        wp.launch(g2p, dim=n_particles,
                  inputs=[particle_x, particle_v, particle_C, particle_F, grid_v,
                          DX, 1.0 / DX, DT, clip_bound])

    # ---- run + record trajectory ----
    trajectory = []
    time_history = []
    com_y_history = []
    min_y_history = []
    extent_history = []   # (x_extent, y_extent, z_extent) per saved frame
    stretch_history = []  # mean |singular_value(F) - 1| per saved frame -- 0 = undeformed

    for step in range(NUM_STEPS):
        simulation_step()
        if step % SAVE_EVERY == 0:
            x_np = particle_x.numpy().copy()
            F_np = particle_F.numpy()
            trajectory.append(x_np)
            time_history.append(step * DT)
            com_y_history.append(float(x_np[:, 1].mean()))
            min_y_history.append(float(x_np[:, 1].min()))
            extent_history.append(tuple((x_np.max(axis=0) - x_np.min(axis=0)).tolist()))
            sigma = np.linalg.svd(F_np, compute_uv=False)  # (n_particles, 3) singular values
            stretch_history.append(float(np.mean(np.abs(sigma - 1.0))))

    wp.synchronize()
    trajectory = np.stack(trajectory)
    extent_arr = np.array(extent_history)  # (num_frames, 3)

    sphere_diameter = 2 * SPHERE_RADIUS
    print(f"\nSimulated {NUM_STEPS} steps = {NUM_STEPS * DT:.3f} s")
    print(f"Final center-of-mass height: {com_y_history[-1]:.4f}")
    print(f"Final minimum particle height: {min_y_history[-1]:.4f} (floor = {floor_y:.3f})")
    print(f"\nOriginal sphere diameter: {sphere_diameter:.4f}")
    print(f"Final bounding-box extents (x, y, z): "
          f"({extent_arr[-1,0]:.4f}, {extent_arr[-1,1]:.4f}, {extent_arr[-1,2]:.4f})")
    print(f"Final mean |singular_value(F) - 1| (0 = undeformed): {stretch_history[-1]:.4f}")
    print(f"Max  mean |singular_value(F) - 1| over the run:      {max(stretch_history):.4f}")

    # ---- shape diagnostics plot ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(time_history, extent_arr[:, 0], label='x extent')
    axes[0].plot(time_history, extent_arr[:, 1], label='y extent (vertical)')
    axes[0].plot(time_history, extent_arr[:, 2], label='z extent')
    axes[0].axhline(sphere_diameter, color='gray', linestyle=':', label='original diameter')
    axes[0].set_xlabel('time (s)'); axes[0].set_ylabel('extent')
    axes[0].set_title('Bounding-box extents over time')
    axes[0].legend(fontsize=8)

    axes[1].plot(time_history, stretch_history, color='tab:red')
    axes[1].set_xlabel('time (s)'); axes[1].set_ylabel('mean |sigma - 1|')
    axes[1].set_title('Deformation magnitude over time\n(0 = undeformed sphere)')

    plt.tight_layout()
    diagnostics_path = os.path.join(OUTPUT_DIR, "shape_diagnostics.png")
    plt.savefig(diagnostics_path, dpi=100)
    plt.close(fig)
    print(f"\nSaved shape diagnostics plot to {diagnostics_path}")

    # ---- render video ----
    render_video(trajectory, time_history, floor_y)

    # ---- floor contact time ----
    min_y_arr = np.array(min_y_history)
    contact_idx = int(np.argmax(min_y_arr <= floor_y + 1e-3))
    contact_time = time_history[contact_idx]

    # ---- validate free fall against the analytical curve, up to first contact ----
    # y(t) = y0 - 1/2 * g * t^2, compared against the simulated center-of-mass height
    time_arr = np.array(time_history)
    com_y_arr = np.array(com_y_history)
    y0 = com_y_arr[0]
    pre_contact = slice(0, max(contact_idx, 1))
    analytical = y0 - 0.5 * abs(GRAVITY[1]) * time_arr[pre_contact] ** 2
    free_fall_error = float(np.max(np.abs(com_y_arr[pre_contact] - analytical)))
    free_fall_validated = bool(free_fall_error < 0.01)  # within 1 grid-cell (dx=0.05) tolerance

    print(f"\nMax free-fall deviation from analytical (pre-contact): {free_fall_error:.4e}")
    print(f"Free-fall validated (< 0.01): {free_fall_validated}")

    # ---- save metadata ----
    metadata = {
        "id": "000_elastic_sphere_free_fall",
        "description": (
            "Solid elastic sphere free-falling under gravity onto a floor "
            "boundary. Standalone MLS-MPM re-implementation of the UniPhy "
            "(CVPR 2025) / NCLaw simulator core, validated against analytical "
            "free fall (g=9.8 m/s^2) and floor non-penetration."
        ),
        "framework": {
            "name": "uniphy_mpm",
            "note": "Custom Warp re-implementation based on UniPhy_CVPR2025 "
                    "(HimangiM/UniPhy_CVPR2025), distinct from the Genesis-based "
                    "pipeline used in earlier simulations.",
            "reference": "https://github.com/HimangiM/UniPhy_CVPR2025",
            "warp_version": wp.config.version,
            "method": "MLS-MPM (P2G -> grid_op -> G2P) with corotated elasticity",
        },
        "scene": {
            "domain": "[0,1]^3 unit cube",
            "num_grids": NUM_GRIDS,
            "dx": DX,
            "dt": DT,
            "num_steps": NUM_STEPS,
            "gravity": GRAVITY.tolist(),
            "bound": BOUND,
            "boundary_condition": "freeslip",
            "floor_y": float(floor_y),
            "eps": EPS,
            "quality_preset": "low (experiments/configs/sim/low.yaml)",
        },
        "object": {
            "shape": "sphere",
            "shape_source": "configs/objects/sphere/geometry/default.yaml (radius=0.1)",
            "center": SPHERE_CENTER.tolist(),
            "radius": SPHERE_RADIUS,
            "resolution": RES,
            "num_particles": int(n_particles),
            "particle_volume": float(particle_vol),
            "particle_mass": float(particle_mass),
            "total_mass_kg": float(particle_mass * n_particles),
        },
        "material": {
            "type": "corotated_elasticity",
            "source": "UniPhy jelly preset: "
                      "experiments/configs/env/blob/material/elasticity/corotated_jelly.yaml "
                      "(E=1e5, nu=0.2) and experiments/configs/env/blob/jelly.yaml (rho=1e3)",
            "mu": MU,
            "lam": LAM,
            "rho": RHO,
            "E_equivalent": float(E),
            "nu_equivalent": float(NU),
        },
        "results": {
            "initial_com_y": com_y_history[0],
            "min_com_y": min(com_y_history),
            "final_com_y": com_y_history[-1],
            "final_min_particle_y": min_y_history[-1],
            "floor_contact_time_s": float(contact_time),
            "free_fall_max_deviation": free_fall_error,
            "free_fall_validated": free_fall_validated,
            "floor_penetration": bool(min(min_y_history) < floor_y - 1e-3),
            "original_diameter": sphere_diameter,
            "final_extents_xyz": [float(v) for v in extent_arr[-1]],
            "max_extent_xyz": [float(v) for v in extent_arr.max(axis=0)],
            "final_deformation_stretch": stretch_history[-1],
            "max_deformation_stretch": max(stretch_history),
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {METADATA_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
