import bpy
import numpy as np
import os

# Added for sim 011 (rigid probe poking the elastic tissue sphere). Additive-only
# and self-skipping: if this sim's output dir has no probe_keyframes.npy (i.e.
# every other sim, 001-005/010), this script does nothing, so it's safe to leave
# in the shared scripts/ folder and include in any render run without touching
# the proven pipeline for the other sims. Slots into the run order as 07 (that
# numeric gap has been unused since the old wall_decoration/surgical_instruments
# scripts were deleted -- see CLAUDE.md).

sim_output_dir = os.environ.get(
    "SIM_OUTPUT_DIR",
    "/home/elec594/Desktop/surgical-robotics-tissue-sim/simulation/output/001_elastic_sphere_tissue_free_fall_test",
)
probe_path = f"{sim_output_dir}/probe_keyframes.npy"

for obj_name in list(bpy.data.objects.keys()):
    if obj_name.startswith("ProbeSphere"):
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

if not os.path.exists(probe_path):
    print(f"No probe_keyframes.npy at {probe_path} -- skipping probe animation (not a probe-poke sim).")
else:
    data = np.load(probe_path)  # shape (40, 3): x, y, z (sim coordinates)
    PROBE_RADIUS = 0.06  # must match PROBE_RADIUS in dataset_generate_011_probe_poke.py

    bpy.ops.mesh.primitive_uv_sphere_add(radius=PROBE_RADIUS, segments=32, ring_count=16, location=(0, 0, 0))
    probe = bpy.context.active_object
    probe.name = "ProbeSphere"
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new("ProbeMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.78, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.9
        bsdf.inputs["Roughness"].default_value = 0.25
    probe.data.materials.append(mat)

    scene = bpy.context.scene
    # Same coordinate remap as 01_animate_sphere.py: sim_x -> blender_x,
    # sim_z -> blender_y, sim_y -> blender_z (sim Y is up).
    for i, row in enumerate(data):
        x, y, z = row
        frame = i + 1
        scene.frame_set(frame)
        probe.location = (x, z, y)
        probe.keyframe_insert(data_path="location", frame=frame)

    scene.frame_set(1)
    print(f"Probe animated across {data.shape[0]} frames from {probe_path}")
