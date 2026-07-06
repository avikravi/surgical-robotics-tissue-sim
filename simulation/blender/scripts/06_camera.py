import bpy
from mathutils import Vector

# --- Clean up existing custom camera and target (safe re-run) ---
for obj_name in ("HeroCamera", "CameraTarget"):
    if obj_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

TABLE_CENTER_XY = (0.55, 0.0)
SPHERE_TOP_Z = 0.648      # top of sphere at frame 1
TABLE_SURFACE_Z = 0.0164  # sphere resting bottom edge
AIM_Z = TABLE_SURFACE_Z + (SPHERE_TOP_Z - TABLE_SURFACE_Z) * 0.4
TABLE_CENTER = Vector((TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], AIM_Z))

cam_data = bpy.data.cameras.new("HeroCamera")
cam_data.lens = 35
cam_obj = bpy.data.objects.new("HeroCamera", cam_data)
bpy.context.collection.objects.link(cam_obj)

# Balanced wide framing — shows sphere/table/tray plus room context (incl. wall decoration)
cam_obj.location = (TABLE_CENTER_XY[0] - 1.1, TABLE_CENTER_XY[1] - 1.3, AIM_Z + 0.65)

target = bpy.data.objects.new("CameraTarget", None)
target.location = TABLE_CENTER
bpy.context.collection.objects.link(target)
target.empty_display_size = 0.05

constraint = cam_obj.constraints.new(type='TRACK_TO')
constraint.target = target
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

bpy.context.scene.camera = cam_obj

print(f"HeroCamera set to balanced wide framing, aim height={AIM_Z:.3f}m")
