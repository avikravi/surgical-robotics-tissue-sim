import bpy

scene = bpy.context.scene
scene.render.engine = 'CYCLES'

prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences
cycles_prefs.compute_device_type = 'OPTIX'
for device in cycles_prefs.devices:
    device.use = True
scene.cycles.device = 'GPU'

scene.cycles.samples = 256
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPTIX'
scene.cycles.max_bounces = 8
scene.cycles.diffuse_bounces = 4
scene.cycles.glossy_bounces = 4

scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'None'
scene.view_settings.exposure = -0.5  # reduced for stronger shadow definition
scene.view_settings.gamma = 1.0

scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100

# World background: soft ambient fill, reduced strength to preserve shadow contrast
world = scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    scene.world = world
world.use_nodes = True
wnodes = world.node_tree.nodes
wlinks = world.node_tree.links
wnodes.clear()

bg = wnodes.new("ShaderNodeBackground")
bg.location = (0, 0)
bg.inputs['Color'].default_value = (0.65, 0.68, 0.72, 1.0)
bg.inputs['Strength'].default_value = 0.15  # reduced from 0.4 to preserve shadow definition

wout = wnodes.new("ShaderNodeOutputWorld")
wout.location = (300, 0)
wlinks.new(bg.outputs['Background'], wout.inputs['Surface'])

print("Render settings configured: Cycles + OptiX, 256 samples, denoising on, AgX, reduced ambient/exposure for shadow definition")
