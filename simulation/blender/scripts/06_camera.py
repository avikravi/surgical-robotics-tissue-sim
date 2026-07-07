import bpy
from mathutils import Vector

# --- Clean up existing custom camera and target (safe re-run) ---
for obj_name in ("HeroCamera", "CameraTarget"):
    if obj_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

TABLE_CENTER_XY = (0.55, 0.0)
SPHERE_TOP_Z = 0.648      # top of sphere at frame 1
TABLE_SURFACE_Z = 0.0164  # sphere resting bottom edge
AIM_Z = TABLE_SURFACE_Z + (SPHERE_TOP_Z - TABLE_SURFACE_Z) * 0.33
TABLE_CENTER = Vector((TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], AIM_Z))

# DIST_SCALE and LENS solved numerically (see camera-framing analysis) so the sphere reaches
# ~24% of frame height at frame 1 (its closest point to camera, depth-checked across all 40
# frames — the vertical extent actually shrinks during floor impact, so frame 1 is the peak,
# not the squash), while the table and the sphere's full fall stay inside frame at every frame
# with a ~4% safety margin. Going further (larger DIST_SCALE + longer LENS) only asymptotically
# approaches ~25% and requires an increasingly flat/telephoto lens (150mm+) for negligible gain —
# this is close to the practical ceiling for this viewing angle without cropping the table or the
# sphere's early ascent-frame position.
DIST_SCALE = 1.65
LENS = 70.0

cam_data = bpy.data.cameras.new("HeroCamera")
cam_data.lens = LENS
cam_obj = bpy.data.objects.new("HeroCamera", cam_data)
bpy.context.collection.objects.link(cam_obj)

# Tighter/longer-lens framing than the original "wide" setup — prioritizes sphere frame-height
# over showing full room context.
cam_obj.location = (
    TABLE_CENTER_XY[0] - 1.1 * DIST_SCALE,
    TABLE_CENTER_XY[1] - 1.3 * DIST_SCALE,
    AIM_Z + 0.65 * DIST_SCALE,
)

target = bpy.data.objects.new("CameraTarget", None)
target.location = TABLE_CENTER
bpy.context.collection.objects.link(target)
target.empty_display_size = 0.05

constraint = cam_obj.constraints.new(type='TRACK_TO')
constraint.target = target
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

bpy.context.scene.camera = cam_obj

print(f"HeroCamera set to ~24%-frame-height sphere framing, aim height={AIM_Z:.3f}m, lens={LENS}mm")
