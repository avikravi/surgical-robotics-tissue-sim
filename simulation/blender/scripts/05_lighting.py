import bpy
import math

# --- Clean up existing lights (safe re-run) ---
for obj_name in list(bpy.data.objects.keys()):
    if obj_name.startswith(("KeyLight", "FillLight", "RimLight")):
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

default_light = bpy.data.objects.get("Light")
if default_light and default_light.data.type == 'POINT':
    bpy.data.objects.remove(default_light, do_unlink=True)

TABLE_CENTER = (0.55, 0.0, 0.02)

def add_area_light(name, location, rotation_euler, power, size, color=(1.0, 1.0, 1.0)):
    light_data = bpy.data.lights.new(name=name, type='AREA')
    light_data.energy = power
    light_data.size = size
    light_data.color = color
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = location
    light_obj.rotation_euler = rotation_euler
    return light_obj

# Key light — reduced from initial 400W to avoid overexposure
key = add_area_light(
    "KeyLight",
    location=(TABLE_CENTER[0] + 1.2, TABLE_CENTER[1] - 1.2, 1.8),
    rotation_euler=(math.radians(55), 0, math.radians(45)),
    power=180,
    size=0.6,
    color=(1.0, 0.98, 0.95)
)

# Fill light
fill = add_area_light(
    "FillLight",
    location=(TABLE_CENTER[0] - 1.2, TABLE_CENTER[1] + 1.0, 1.4),
    rotation_euler=(math.radians(60), 0, math.radians(-135)),
    power=60,
    size=1.0,
    color=(0.95, 0.97, 1.0)
)

# Rim light
rim = add_area_light(
    "RimLight",
    location=(TABLE_CENTER[0] + 0.2, TABLE_CENTER[1] + 1.8, 1.2),
    rotation_euler=(math.radians(75), 0, math.radians(200)),
    power=100,
    size=0.4,
    color=(1.0, 1.0, 1.0)
)

print("Three-point lighting rig added: KeyLight, FillLight, RimLight (tuned to avoid overexposure)")
