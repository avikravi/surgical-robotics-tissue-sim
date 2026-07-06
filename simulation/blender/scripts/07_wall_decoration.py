import bpy

def clear_material(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat

for obj_name in ("FrameBorder", "FrameCanvas"):
    if obj_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)

# Positioned clear of KeyLight/FillLight/RimLight, on the back wall
FRAME_X = -1.2
FRAME_Z = 1.5
WALL_Y = 2.0 - 0.02  # just in front of back wall (ROOM_SIZE/2 - small offset)

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(FRAME_X, WALL_Y, FRAME_Z))
frame = bpy.context.active_object
frame.name = "FrameBorder"
frame.scale = (0.5, 0.02, 0.65)

frame_mat = clear_material("FrameWood")
nt = frame_mat.node_tree
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
output = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf.inputs['Base Color'].default_value = (0.15, 0.10, 0.06, 1.0)
bsdf.inputs['Roughness'].default_value = 0.4
nt.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
frame.data.materials.append(frame_mat)

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(FRAME_X, WALL_Y - 0.015, FRAME_Z))
canvas = bpy.context.active_object
canvas.name = "FrameCanvas"
canvas.scale = (0.42, 0.005, 0.55)

canvas_mat = clear_material("CanvasPrint")
nt = canvas_mat.node_tree
nodes, links = nt.nodes, nt.links
output = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
tex_coord = nodes.new("ShaderNodeTexCoord")
mapping = nodes.new("ShaderNodeMapping")
gradient = nodes.new("ShaderNodeTexGradient")
gradient.gradient_type = 'LINEAR'
ramp = nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.05, 0.25, 0.35, 1.0)
ramp.color_ramp.elements[1].color = (0.45, 0.55, 0.30, 1.0)

links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], gradient.inputs['Vector'])
links.new(gradient.outputs['Fac'], ramp.inputs['Fac'])
links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
bsdf.inputs['Roughness'].default_value = 0.6
canvas.data.materials.append(canvas_mat)

print("Wall picture frame decoration added, positioned clear of light rig")
