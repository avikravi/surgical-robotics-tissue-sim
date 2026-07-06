import bpy

OBJECT_NAME = "TissueSphere"
MAT_NAME = "TissueMaterial"

obj = bpy.data.objects.get(OBJECT_NAME)
if obj is None:
    raise RuntimeError(f"Object '{OBJECT_NAME}' not found. Run 01_animate_sphere.py first.")

mat = bpy.data.materials.get(MAT_NAME)
if mat is None:
    mat = bpy.data.materials.new(MAT_NAME)
mat.use_nodes = True
nt = mat.node_tree
nodes = nt.nodes
links = nt.links
nodes.clear()

output = nodes.new("ShaderNodeOutputMaterial"); output.location = (600, 0)
bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
tex_coord = nodes.new("ShaderNodeTexCoord"); tex_coord.location = (-800, 0)
mapping = nodes.new("ShaderNodeMapping"); mapping.location = (-600, 0)
mapping.inputs['Scale'].default_value = (1, 1, 1)

# Base color blotching (subtle)
noise_color = nodes.new("ShaderNodeTexNoise")
noise_color.location = (-400, 300)
noise_color.inputs['Scale'].default_value = 4.0
noise_color.inputs['Detail'].default_value = 4.0
noise_color.inputs['Roughness'].default_value = 0.5

ramp_color = nodes.new("ShaderNodeValToRGB")
ramp_color.location = (-150, 300)
ramp_color.color_ramp.elements[0].position = 0.4
ramp_color.color_ramp.elements[0].color = (0.60, 0.20, 0.18, 1.0)
ramp_color.color_ramp.elements[1].position = 0.6
ramp_color.color_ramp.elements[1].color = (0.78, 0.42, 0.38, 1.0)

# Roughness variation (subtle)
noise_rough = nodes.new("ShaderNodeTexNoise")
noise_rough.location = (-400, 0)
noise_rough.inputs['Scale'].default_value = 6.0
noise_rough.inputs['Detail'].default_value = 2.0

ramp_rough = nodes.new("ShaderNodeValToRGB")
ramp_rough.location = (-150, 0)
ramp_rough.color_ramp.elements[0].position = 0.35
ramp_rough.color_ramp.elements[0].color = (0.35, 0.35, 0.35, 1.0)
ramp_rough.color_ramp.elements[1].position = 0.65
ramp_rough.color_ramp.elements[1].color = (0.50, 0.50, 0.50, 1.0)

links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], noise_color.inputs['Vector'])
links.new(noise_color.outputs['Fac'], ramp_color.inputs['Fac'])
links.new(ramp_color.outputs['Color'], bsdf.inputs['Base Color'])

links.new(mapping.outputs['Vector'], noise_rough.inputs['Vector'])
links.new(noise_rough.outputs['Fac'], ramp_rough.inputs['Fac'])
links.new(ramp_rough.outputs['Color'], bsdf.inputs['Roughness'])

links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

# Mild subsurface scattering for organic look
bsdf.inputs['Subsurface Weight'].default_value = 0.2
bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.4, 0.3)
bsdf.inputs['Subsurface Scale'].default_value = 0.1
bsdf.inputs['Specular IOR Level'].default_value = 0.5

if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)

print(f"Tissue material '{MAT_NAME}' applied to '{obj.name}' — no bump, sphere shape preserved.")
