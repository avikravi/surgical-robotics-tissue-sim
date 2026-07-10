import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("sim_dir", nargs="?", default="output/001_elastic_sphere_tissue_free_fall_test")
parser.add_argument("--num-frames", type=int, default=2000)
parser.add_argument("--num-samples", type=int, default=40)
args = parser.parse_args()

SIM_DIR = args.sim_dir
NUM_FRAMES = args.num_frames
NUM_SAMPLES = args.num_samples

x = torch.load(f"{SIM_DIR}/GtX.pt", map_location="cpu").numpy()
x = x[:NUM_FRAMES]

indices = np.linspace(0, NUM_FRAMES - 1, NUM_SAMPLES).round().astype(int)

keyframes = np.zeros((NUM_SAMPLES, 6), dtype=np.float32)
for row, i in enumerate(indices):
    frame = x[i]
    center = frame.mean(axis=0)
    extent = frame.max(axis=0) - frame.min(axis=0)
    keyframes[row, :3] = center
    keyframes[row, 3:] = extent

out_path = f"{SIM_DIR}/sphere_keyframes.npy"
np.save(out_path, keyframes)

print(f"Saved {keyframes.shape} to {out_path}")
print("first row (center_x,y,z, extent_x,y,z):", keyframes[0])
print("last row  (center_x,y,z, extent_x,y,z):", keyframes[-1])
