import bpy
import numpy as np

# --- Load the keyframe data ---
npy_path = "/home/elec594/Desktop/surgical-robotics-tissue-sim/simulation/output/001_elastic_sphere_tissue_free_fall_test/sphere_keyframes.npy"
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

# --- Add a ground plane ---
bpy.ops.mesh.primitive_plane_add(size=4, location=(0.55, 0.0, 0.0))
plane = bpy.context.active_object
plane.name = "Ground"

scene.frame_set(1)
print("Done — sphere keyframed across", data.shape[0], "frames, scene cleaned of duplicates")
