import bpy
import numpy as np
import os

# --- Load the keyframe data ---
# SIM_OUTPUT_DIR lets a headless batch-render driver point this at any sim's output folder;
# manual use from Blender's Scripting tab (no env var set) still defaults to 001 as before.
sim_output_dir = os.environ.get(
    "SIM_OUTPUT_DIR",
    "/home/elec594/Desktop/surgical-robotics-tissue-sim/simulation/output/001_elastic_sphere_tissue_free_fall_test",
)
npy_path = f"{sim_output_dir}/sphere_keyframes.npy"
data = np.load(npy_path)  # shape (40, 6): center_x,y,z, extent_x,y,z (sim coordinates)

# --- Clean up any existing TissueSphere / Ground / Cube objects first (safe re-run) ---
for obj_name in list(bpy.data.objects.keys()):
    if obj_name.startswith("TissueSphere") or obj_name.startswith("Ground") or obj_name == "Cube":
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

# --- Create a UV sphere for the tissue ---
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(0, 0, 0))
sphere = bpy.context.active_object
sphere.name = "TissueSphere"
bpy.ops.object.shade_smooth()

# --- Coordinate mapping: sim_x -> blender_x, sim_z -> blender_y, sim_y -> blender_z (sim Y is up) ---
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = data.shape[0]

for i, row in enumerate(data):
    cx, cy, cz, ex, ey, ez = row
    frame = i + 1
    scene.frame_set(frame)
    sphere.location = (cx, cz, cy)
    sphere.scale = (ex / 2.0, ez / 2.0, ey / 2.0)
    sphere.keyframe_insert(data_path="location", frame=frame)
    sphere.keyframe_insert(data_path="scale", frame=frame)

# NOTE: this script used to also add a small 4x4 "Ground" placeholder plane here, before
# 03_room_geometry.py's real RoomFloor existed. It was never removed and never given a
# material, so it sat exactly coplanar with RoomFloor (both at z=0) and z-fought with it,
# rendering as a hard black wedge wherever Cycles picked the unmaterialed "Ground" face
# (Cycles renders objects with no material as pure black) instead of RoomFloor's tile
# shader. The cleanup loop above still removes it by name so any older .blend file that
# still has a leftover "Ground" object gets fixed on re-run.

scene.frame_set(1)
print("Done — sphere keyframed across", data.shape[0], "frames, scene cleaned of duplicates")
