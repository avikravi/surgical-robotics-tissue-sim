import argparse
import json
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("sim_dir")
parser.add_argument("--dt", type=float, default=1.0 / 4800)
parser.add_argument("--gravity", type=float, default=9.8)
parser.add_argument("--material-kind", choices=["tensor", "fluid"], default="tensor")
args = parser.parse_args()

x = torch.load(f"{args.sim_dir}/GtX.pt", map_location="cpu").numpy()
F = torch.load(f"{args.sim_dir}/GtF.pt", map_location="cpu").numpy()

# GtX/GtV's last two rows are leftover zero-init buffer slots never written by the save loop
# (dataset_generate.py's loop only fills indices 0..num_sim_steps*substep_gt-1); GtF's index 0
# is likewise never written (its save loop starts at get_F(i+1,...)). Trim both ends safely.
n_valid = x.shape[0] - 2
x = x[:n_valid]
F = F[1:n_valid]

y = x[:, :, 1]
com_y = y.mean(axis=1)
min_y_per_step = y.min(axis=1)

initial_com_y = float(com_y[0])
final_com_y = float(com_y[-1])
min_com_y = float(com_y.min())
min_particle_y = float(min_y_per_step.min())
floor_penetration = bool(min_particle_y < -1e-3)

# Free-fall validation: while no particle has yet touched the floor (min_y_per_step > ~0.02),
# compare COM drop to the analytical g*t^2/2 trajectory (measured from the first saved sample).
touch_idx = np.argmax(min_y_per_step < 0.02) if (min_y_per_step < 0.02).any() else len(min_y_per_step)
free_fall_region = slice(0, max(touch_idx, 1))
t = np.arange(x.shape[0]) * args.dt
analytical_drop = 0.5 * args.gravity * t**2
actual_drop = com_y[0] - com_y
deviation = np.abs(actual_drop[free_fall_region] - analytical_drop[free_fall_region])
free_fall_max_deviation = float(deviation.max()) if len(deviation) else 0.0
floor_contact_time_s = float(t[touch_idx]) if touch_idx < len(t) else None

extents_all = x.max(axis=1) - x.min(axis=1)
final_extents_xyz = extents_all[-1].tolist()
max_extent_xyz = extents_all.max(axis=0).tolist()
original_diameter = float(extents_all[0].mean())

result = dict(
    initial_com_y=round(initial_com_y, 5),
    min_com_y=round(min_com_y, 5),
    final_com_y=round(final_com_y, 5),
    min_particle_y_all_frames=round(min_particle_y, 5),
    floor_penetration=floor_penetration,
    floor_contact_time_s=round(floor_contact_time_s, 5) if floor_contact_time_s is not None else None,
    free_fall_max_deviation=round(free_fall_max_deviation, 5),
    free_fall_validated=bool(free_fall_max_deviation < 0.02),
    original_diameter=round(original_diameter, 4),
    final_extents_xyz=[round(v, 4) for v in final_extents_xyz],
    max_extent_xyz=[round(v, 4) for v in max_extent_xyz],
)

if args.material_kind == "fluid":
    result["final_deformation_stretch"] = None
    result["max_deformation_stretch"] = None
else:
    eye3 = np.eye(3, dtype=np.float32)
    sig = np.linalg.svd(F, compute_uv=False)  # (T,P,3)
    stretch = np.abs(sig - 1.0).mean(axis=1)  # per-step mean over particles, averaged over 3 sing. vals below
    stretch = stretch.mean(axis=1)
    result["final_deformation_stretch"] = round(float(stretch[-1]), 5)
    result["max_deformation_stretch"] = round(float(stretch.max()), 5)
    result["max_deformation_stretch_frame"] = int(stretch.argmax()) + 1

print(json.dumps(result, indent=2))
