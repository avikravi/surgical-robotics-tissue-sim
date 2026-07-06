import bpy
import math

for obj_name in list(bpy.data.objects.keys()):
    if obj_name.startswith(("Scalpel", "Forceps", "InstrumentTray")):
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

def clear_material(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat

steel_mat = clear_material("SurgicalSteel")
nt = steel_mat.node_tree
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
output = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf.inputs['Base Color'].default_value = (0.72, 0.73, 0.75, 1.0)
bsdf.inputs['Metallic'].default_value = 1.0
bsdf.inputs['Roughness'].default_value = 0.25
nt.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# Final position: next to table, within camera view
TOOLS_X = 1.15
TOOLS_Y = -0.25
FLOOR_Z = 0.005

# Scalpel: thin flat blade + cylindrical handle
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(TOOLS_X, TOOLS_Y, FLOOR_Z))
blade = bpy.context.active_object
blade.name = "Scalpel_Blade"
blade.scale = (0.015, 0.05, 0.002)
blade.rotation_euler = (0, 0, math.radians(20))
blade.data.materials.append(steel_mat)

bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=0.12, location=(TOOLS_X - 0.08, TOOLS_Y - 0.03, FLOOR_Z))
handle = bpy.context.active_object
handle.name = "Scalpel_Handle"
handle.rotation_euler = (math.radians(90), 0, math.radians(20))
handle.data.materials.append(steel_mat)

bpy.ops.object.select_all(action='DESELECT')
blade.select_set(True)
handle.select_set(True)
bpy.context.view_layer.objects.active = blade
bpy.ops.object.join()
blade.name = "Scalpel"

# Forceps
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(TOOLS_X + 0.15, TOOLS_Y + 0.1, FLOOR_Z))
prong1 = bpy.context.active_object
prong1.name = "Forceps_Prong1"
prong1.scale = (0.004, 0.08, 0.003)
prong1.rotation_euler = (0, 0, math.radians(8))
prong1.data.materials.append(steel_mat)

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(TOOLS_X + 0.16, TOOLS_Y + 0.1, FLOOR_Z))
prong2 = bpy.context.active_object
prong2.name = "Forceps_Prong2"
prong2.scale = (0.004, 0.08, 0.003)
prong2.rotation_euler = (0, 0, math.radians(-8))
prong2.data.materials.append(steel_mat)

bpy.ops.object.select_all(action='DESELECT')
prong1.select_set(True)
prong2.select_set(True)
bpy.context.view_layer.objects.active = prong1
bpy.ops.object.join()
prong1.name = "Forceps"

# Instrument tray
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(TOOLS_X + 0.05, TOOLS_Y - 0.15, FLOOR_Z))
tray = bpy.context.active_object
tray.name = "InstrumentTray"
tray.scale = (0.18, 0.12, 0.004)
tray.data.materials.append(steel_mat)

print("Surgical instruments positioned next to table, within camera view")
