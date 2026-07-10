import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("sim_dir")
parser.add_argument("out_png")
parser.add_argument("--title", default="")
parser.add_argument("--material-kind", choices=["tensor", "fluid"], default="tensor",
                     help="'tensor' materials (elasticity/von_mises/drucker_prager) track a real "
                          "F; 'fluid' (viscous_fluid) only tracks a scalar volume ratio in F[...,0,0].")
parser.add_argument("--num-frames", type=int, default=2000)
parser.add_argument("--num-samples", type=int, default=40)
args = parser.parse_args()

x = torch.load(f"{args.sim_dir}/GtX.pt", map_location="cpu").numpy()[:args.num_frames]
F = torch.load(f"{args.sim_dir}/GtF.pt", map_location="cpu").numpy()[:args.num_frames]

indices = np.linspace(0, args.num_frames - 1, args.num_samples).round().astype(int)

extents = np.zeros((args.num_samples, 3), dtype=np.float32)
deformation = np.zeros(args.num_samples, dtype=np.float32)

eye3 = np.eye(3, dtype=np.float32)
for row, i in enumerate(indices):
    frame_x = x[i]
    extents[row] = frame_x.max(axis=0) - frame_x.min(axis=0)
    # GtF index 0 is never written by dataset_generate.py's save loop (leftover zero-init
    # memory, not real physics -- see CLAUDE.md's dt/frame gotcha note), so never sample it.
    frame_F = F[max(i, 1)]
    if args.material_kind == "fluid":
        # Only F[...,0,0] is a meaningful scalar volume ratio J for this material; deviation
        # from J=1 (no volume change) is the analogous "deformation magnitude".
        deformation[row] = np.abs(frame_F[:, 0, 0] - 1.0).mean()
    else:
        deformation[row] = np.linalg.norm(frame_F - eye3, axis=(1, 2)).mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

frames = np.arange(1, args.num_samples + 1)
axes[0].plot(frames, extents[:, 0], label="extent_x", color="#f5a742")
axes[0].plot(frames, extents[:, 1], label="extent_y (vertical)", color="#1fab89")
axes[0].plot(frames, extents[:, 2], label="extent_z", color="#8a7fe0")
axes[0].set_title("Bounding-Box Extents Over Time")
axes[0].set_xlabel("Frame")
axes[0].set_ylabel("Extent (sim units)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(frames, deformation, color="#2fd7c4")
if args.material_kind == "fluid":
    axes[1].set_title(f"{args.title} Volume-Ratio Deviation |J-1| Over Time")
    axes[1].set_ylabel("Mean |J-1| (dimensionless)")
else:
    axes[1].set_title(f"{args.title} Deformation Magnitude Over Time")
    axes[1].set_ylabel("Mean ||F-I|| (sim units)")
axes[1].set_xlabel("Frame")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(args.out_png, dpi=110)
print(f"Saved diagnostics to {args.out_png}")
