import bpy

# --- Clean up any existing room objects first (safe re-run) ---
for obj_name in list(bpy.data.objects.keys()):
    if obj_name.startswith(("RoomFloor", "RoomWall", "Table")):
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

ROOM_SIZE = 4.0
ROOM_HEIGHT = 3.0
TABLE_HEIGHT = 0.0164  # matches sphere's resting bottom edge (Location Z 0.0584 - Scale Z 0.042)

# --- Floor ---
bpy.ops.mesh.primitive_plane_add(size=ROOM_SIZE, location=(0, 0, 0))
floor = bpy.context.active_object
floor.name = "RoomFloor"

# --- Back wall ---
bpy.ops.mesh.primitive_plane_add(size=ROOM_SIZE)
wall_back = bpy.context.active_object
wall_back.name = "RoomWall_Back"
wall_back.rotation_euler = (1.5708, 0, 0)
wall_back.location = (0, ROOM_SIZE / 2, ROOM_HEIGHT / 2)
wall_back.scale = (1, 1, ROOM_HEIGHT / ROOM_SIZE)

# --- Left wall ---
bpy.ops.mesh.primitive_plane_add(size=ROOM_SIZE)
wall_left = bpy.context.active_object
wall_left.name = "RoomWall_Left"
wall_left.rotation_euler = (1.5708, 0, 1.5708)
wall_left.location = (-ROOM_SIZE / 2, 0, ROOM_HEIGHT / 2)
wall_left.scale = (1, 1, ROOM_HEIGHT / ROOM_SIZE)

# --- Table (thin slab, top surface at sphere's resting height) ---
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.55, 0.0, TABLE_HEIGHT / 2))
table = bpy.context.active_object
table.name = "Table"
table.scale = (0.8, 0.5, TABLE_HEIGHT / 2)

bpy.ops.object.select_all(action='DESELECT')
for o in (floor, wall_back, wall_left, table):
    o.select_set(True)
bpy.context.view_layer.objects.active = table
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

print("Room + table geometry created, table height matched to sphere rest position")
