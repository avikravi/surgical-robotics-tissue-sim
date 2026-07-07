import bpy

def clear_material(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat

# =========================================================
# WOOD MATERIAL (Table)
# =========================================================
wood_mat = clear_material("WoodTable")
nt = wood_mat.node_tree
nodes, links = nt.nodes, nt.links

output = nodes.new("ShaderNodeOutputMaterial"); output.location = (900, 0)
bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (600, 0)
tex_coord = nodes.new("ShaderNodeTexCoord"); tex_coord.location = (-900, 0)
mapping = nodes.new("ShaderNodeMapping"); mapping.location = (-700, 0)
mapping.inputs['Scale'].default_value = (4.0, 1.2, 1.0)

wave = nodes.new("ShaderNodeTexWave")
wave.location = (-500, 250)
wave.wave_type = 'BANDS'
wave.inputs['Scale'].default_value = 10.0
wave.inputs['Distortion'].default_value = 3.5
wave.inputs['Detail'].default_value = 4.0
wave.inputs['Detail Scale'].default_value = 2.0

noise_color = nodes.new("ShaderNodeTexNoise")
noise_color.location = (-500, -50)
noise_color.inputs['Scale'].default_value = 6.0
noise_color.inputs['Detail'].default_value = 5.0
noise_color.inputs['Roughness'].default_value = 0.65

mix_grain = nodes.new("ShaderNodeMixRGB")
mix_grain.location = (-250, 100)
mix_grain.blend_type = 'MULTIPLY'
mix_grain.inputs['Fac'].default_value = 0.45

ramp_wood = nodes.new("ShaderNodeValToRGB")
ramp_wood.location = (0, 200)
ramp_wood.color_ramp.elements[0].position = 0.25
ramp_wood.color_ramp.elements[0].color = (0.18, 0.09, 0.03, 1.0)
ramp_wood.color_ramp.elements.new(0.55)
ramp_wood.color_ramp.elements[1].color = (0.52, 0.30, 0.14, 1.0)
ramp_wood.color_ramp.elements[2].position = 0.8
ramp_wood.color_ramp.elements[2].color = (0.68, 0.44, 0.22, 1.0)

bump_wood = nodes.new("ShaderNodeBump")
bump_wood.location = (250, -200)
bump_wood.inputs['Strength'].default_value = 0.15
bump_wood.inputs['Distance'].default_value = 0.01

links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], wave.inputs['Vector'])
links.new(mapping.outputs['Vector'], noise_color.inputs['Vector'])
links.new(wave.outputs['Fac'], mix_grain.inputs['Color1'])
links.new(noise_color.outputs['Fac'], mix_grain.inputs['Color2'])
links.new(mix_grain.outputs['Color'], ramp_wood.inputs['Fac'])
links.new(ramp_wood.outputs['Color'], bsdf.inputs['Base Color'])
links.new(wave.outputs['Fac'], bump_wood.inputs['Height'])
links.new(bump_wood.outputs['Normal'], bsdf.inputs['Normal'])
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
bsdf.inputs['Roughness'].default_value = 0.4
bsdf.inputs['Specular IOR Level'].default_value = 0.5

# =========================================================
# WALL MATERIAL (light blue, subtle procedural variation)
# =========================================================
wall_mat = clear_material("WallPaint")
nt = wall_mat.node_tree
nodes, links = nt.nodes, nt.links

output = nodes.new("ShaderNodeOutputMaterial"); output.location = (700, 0)
bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (400, 0)
tex_coord = nodes.new("ShaderNodeTexCoord"); tex_coord.location = (-800, 0)
mapping = nodes.new("ShaderNodeMapping"); mapping.location = (-600, 0)
mapping.inputs['Scale'].default_value = (1, 1, 1)

noise_rough = nodes.new("ShaderNodeTexNoise")
noise_rough.location = (-400, 200)
noise_rough.inputs['Scale'].default_value = 25.0
noise_rough.inputs['Detail'].default_value = 3.0

ramp_rough = nodes.new("ShaderNodeValToRGB")
ramp_rough.location = (-150, 200)
ramp_rough.color_ramp.elements[0].position = 0.4
ramp_rough.color_ramp.elements[0].color = (0.62, 0.75, 0.86, 1.0)
ramp_rough.color_ramp.elements[1].position = 0.6
ramp_rough.color_ramp.elements[1].color = (0.78, 0.88, 0.96, 1.0)

noise_bump = nodes.new("ShaderNodeTexNoise")
noise_bump.location = (-400, -200)
noise_bump.inputs['Scale'].default_value = 40.0
noise_bump.inputs['Detail'].default_value = 2.0

bump_wall = nodes.new("ShaderNodeBump")
bump_wall.location = (-150, -200)
bump_wall.inputs['Strength'].default_value = 0.08
bump_wall.inputs['Distance'].default_value = 0.002

links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], noise_rough.inputs['Vector'])
links.new(noise_rough.outputs['Fac'], ramp_rough.inputs['Fac'])
links.new(ramp_rough.outputs['Color'], bsdf.inputs['Base Color'])
links.new(ramp_rough.outputs['Color'], bsdf.inputs['Roughness'])
links.new(mapping.outputs['Vector'], noise_bump.inputs['Vector'])
links.new(noise_bump.outputs['Fac'], bump_wall.inputs['Height'])
links.new(bump_wall.outputs['Normal'], bsdf.inputs['Normal'])
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
bsdf.inputs['Specular IOR Level'].default_value = 0.1

# =========================================================
# FLOOR MATERIAL (tile, distinct from wood table)
# =========================================================
floor_mat = clear_material("FloorTile")
nt = floor_mat.node_tree
nodes, links = nt.nodes, nt.links

output = nodes.new("ShaderNodeOutputMaterial"); output.location = (900, 0)
bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (600, 0)
tex_coord = nodes.new("ShaderNodeTexCoord"); tex_coord.location = (-900, 0)
mapping = nodes.new("ShaderNodeMapping"); mapping.location = (-700, 0)
# Object-space coords are real meters (RoomFloor's scale is baked in via transform_apply),
# so tiles_per_meter = mapping.Scale * brick.Scale exactly, independent of ROOM_SIZE.
# 3.5 -> ~28.6cm tiles (a common real floor-tile size), down from 6.0 (~16.7cm) — coarser
# margin against aliasing now that ROOM_SIZE=9.0 puts far more tile repeats in frame at once
# than the original ROOM_SIZE=4.0 did.
mapping.inputs['Scale'].default_value = (3.5, 3.5, 1.0)

brick = nodes.new("ShaderNodeTexBrick")
brick.location = (-500, 200)
brick.inputs['Color1'].default_value = (0.72, 0.71, 0.68, 1.0)
brick.inputs['Color2'].default_value = (0.68, 0.67, 0.64, 1.0)
brick.inputs['Mortar'].default_value = (0.35, 0.35, 0.34, 1.0)
brick.inputs['Scale'].default_value = 1.0
brick.inputs['Mortar Size'].default_value = 0.015
brick.inputs['Row Height'].default_value = 1.0
brick.offset = 0.0

noise_floor = nodes.new("ShaderNodeTexNoise")
noise_floor.location = (-500, -150)
noise_floor.inputs['Scale'].default_value = 15.0
noise_floor.inputs['Detail'].default_value = 3.0

ramp_variation = nodes.new("ShaderNodeValToRGB")
ramp_variation.location = (-250, -150)
ramp_variation.color_ramp.elements[0].position = 0.4
ramp_variation.color_ramp.elements[0].color = (0.9, 0.9, 0.9, 1.0)
ramp_variation.color_ramp.elements[1].position = 0.6
ramp_variation.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

mix_final = nodes.new("ShaderNodeMixRGB")
mix_final.location = (100, 100)
mix_final.blend_type = 'MULTIPLY'
mix_final.inputs['Fac'].default_value = 0.3

bump_floor = nodes.new("ShaderNodeBump")
bump_floor.location = (300, -200)
# Strength reduced and Distance rescaled to keep the same ~3% of tile-period ratio as before,
# now that the tile period grew from ~0.167m to ~0.286m (see mapping.Scale above) — an
# unscaled Distance would under-sample the new, larger tile features.
bump_floor.inputs['Strength'].default_value = 0.15
bump_floor.inputs['Distance'].default_value = 0.0086

links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], brick.inputs['Vector'])
links.new(mapping.outputs['Vector'], noise_floor.inputs['Vector'])
links.new(noise_floor.outputs['Fac'], ramp_variation.inputs['Fac'])
links.new(brick.outputs['Color'], mix_final.inputs['Color1'])
links.new(ramp_variation.outputs['Color'], mix_final.inputs['Color2'])
links.new(mix_final.outputs['Color'], bsdf.inputs['Base Color'])
links.new(brick.outputs['Fac'], bump_floor.inputs['Height'])
links.new(bump_floor.outputs['Normal'], bsdf.inputs['Normal'])
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
bsdf.inputs['Roughness'].default_value = 0.45
bsdf.inputs['Specular IOR Level'].default_value = 0.4

# =========================================================
# ASSIGN MATERIALS
# =========================================================
def assign(obj_name, mat):
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print(f"Warning: object '{obj_name}' not found, skipping.")
        return
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

assign("Table", wood_mat)
assign("RoomWall_Back", wall_mat)
assign("RoomWall_Left", wall_mat)
assign("RoomFloor", floor_mat)

print("Materials assigned: WoodTable -> Table, WallPaint (light blue) -> walls, FloorTile -> floor")
