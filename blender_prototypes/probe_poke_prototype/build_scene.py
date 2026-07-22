"""
Blender-NATIVE prototype scene for the rigid-probe-poke interaction (sim 006 concept).

*** THIS USES BLENDER'S BUILT-IN SOFT BODY (MASS-SPRING) PHYSICS SYSTEM ***
*** THIS IS NOT THE REAL UNIPHY / MLS-MPM SOLVER USED FOR SIMS 001-005  ***

Soft Body is a mass-spring system solved on mesh edges with an optional "goal"
pull-back-to-rest-shape term -- a completely different numerical model from the
Taichi MLS-MPM material point method used for the real dataset (no material
points, no grid, no constitutive stress law, no Lame/yield/friction parameters).
This script exists ONLY to sanity-check the geometry, motion, and visual feel of
a probe-poke interaction before implementing it properly in Taichi MPM. Its
results are qualitative-only and must never be compared numerically (deformation
magnitude, timing, contact forces, anything quantitative) against the real
GtX/GtF/GtStress tensors or dataset_viewer.html entries for sims 001-005. There
is no shared code, data, or config with simulation/, dataset_viewer.html, or
experiments/proxy_net/ -- this whole prototype is self-contained under
blender_prototypes/probe_poke_prototype/.

Run headless to (re)generate the .blend file:
    blender --background --python build_scene.py

Physics is left UNBAKED and nothing is rendered here on purpose (per the
experiment brief) -- open probe_poke_prototype.blend in the Blender GUI, tweak
Soft Body settings on the "Blob" object if needed, bake (Physics tab > Soft
Body > Cache > Bake), and scrub/render interactively from there.

All key numbers (blob size/deformation, probe size/timing, Soft Body settings)
are defined as named constants right below and are also printed to stdout when
this script runs, so the exact values baked into the saved .blend file are
always verifiable from the console output, not just this source file.
"""

import os

import bpy
from mathutils import noise

# ── Blob (soft body) geometry ────────────────────────────────────────────────
BLOB_RADIUS = 0.5          # icosphere radius before squash, in meters
BLOB_SUBDIV = 3            # icosphere subdivision level (162 verts, verified) --
                            # enough smoothness for a soft body sim without
                            # being too heavy to simulate interactively
BLOB_SQUASH_Z = 0.85       # non-uniform z-scale applied to the icosphere so it
                            # reads as an organic blob resting under its own
                            # weight, not a perfect sphere
BLOB_NOISE_SCALE = 2.5     # spatial frequency of the per-vertex Perlin noise
                            # displacement (higher = smaller/more numerous bumps)
BLOB_NOISE_AMPLITUDE = 0.035  # +/- displacement along each vertex normal, in
                            # meters (~7% of BLOB_RADIUS) -- gentle, not
                            # distorting the overall blob shape

BLOB_HALF_HEIGHT = BLOB_RADIUS * BLOB_SQUASH_Z   # 0.425 m, nominal (pre-noise)
BLOB_HEIGHT = 2 * BLOB_HALF_HEIGHT               # 0.85 m, nominal -- used only
                                                  # as a reference for probe
                                                  # indentation depth below; the
                                                  # blob's actual object
                                                  # placement is computed from
                                                  # its real post-noise geometry
                                                  # (see build_blob()), not this
                                                  # nominal constant, so ground
                                                  # contact is exact regardless
                                                  # of the noise seed

# ── Soft Body settings (Blender mass-spring solver) ─────────────────────────
# Verified Blender 5.1.2 defaults (printed via a throwaway probe script before
# writing this one): mass=1.0, friction=0.5, use_goal=True, goal_default=0.7,
# goal_spring=0.5, goal_friction=0.0, pull=push=0.5, bend=0.0, damping=0.5,
# use_self_collision=False. Only two values are deliberately changed from those
# defaults below (reasoning inline); everything else is left at Blender's
# verified default.
SB_MASS = 1.0               # kg (Blender default -- reasonable for this scale)
SB_GOAL_DEFAULT = 0.3       # lowered from Blender's default 0.7: the default
                            # pulls too strongly back toward the rest shape,
                            # visibly resisting the probe's indentation; 0.3
                            # still prevents total collapse/runaway deformation
                            # but lets the poke actually show

# Root cause of the whole blob drifting/sliding under the probe instead of
# staying anchored while only the poked region deforms: a flat goal_default
# with no vertex-group pinning leaves the base just as free to sag/slide as
# the rest of the mesh. BASE_PIN_HEIGHT_FRACTION is the bottom fraction of the
# blob's actual (post-noise) height that gets pinned via a vertex group at
# BASE_PIN_GOAL_WEIGHT (~1.0, effectively anchored); everything above that
# keeps SB_GOAL_DEFAULT so the poked region stays freely deformable.
BASE_PIN_HEIGHT_FRACTION = 0.15
BASE_PIN_GOAL_WEIGHT = 0.98   # not a full 1.0 so the Soft Body solver still
                              # has a hair of give at the base (avoids a
                              # perfectly rigid seam against the freely-moving
                              # region right above it) -- raise to 1.0 if you
                              # want the base fully locked once you're baking
SB_DAMPING = 5.0            # raised from Blender's default 0.5: with an
                            # animated collision probe pushing hard into the
                            # mesh, the default damping under-damps and risks
                            # sustained jitter/instability ("explode"); 5.0 is
                            # a safer starting point per the "doesn't explode"
                            # requirement -- lower it back down if you want more
                            # jelly-like wobble once you're baking interactively

# ── Probe (rigid, keyframed, Collision physics) ─────────────────────────────
PROBE_RADIUS = 0.15                              # meters (30% of blob radius)
PROBE_CLEARANCE = 0.3                            # gap above blob top at rest, m
PROBE_INDENT_FRACTION = 0.25                      # fraction of BLOB_HEIGHT to
                                                   # indent -- inside the
                                                   # requested 20-30% range
PROBE_INDENT_DEPTH = PROBE_INDENT_FRACTION * BLOB_HEIGHT   # ~0.2125 m

PROBE_START_Z = BLOB_HEIGHT + PROBE_CLEARANCE + PROBE_RADIUS
PROBE_INDENT_Z = (BLOB_HEIGHT - PROBE_INDENT_DEPTH) + PROBE_RADIUS
PROBE_XY = (0.0, 0.0)

# ── Timing ───────────────────────────────────────────────────────────────────
FPS = 24                # matches the fps used for the real UniPhy Blender renders
FRAME_START = 1
FRAME_DESCENT_END = 40    # steady linear descent, frames 1 -> 40 (~1.6s)
FRAME_HOLD_END = 60       # held at full indentation, frames 40 -> 60 (~0.8s)
FRAME_END = 100           # steady linear withdrawal, frames 60 -> 100 (~1.7s)

GROUND_SIZE = 6.0  # meters, square ground plane, visual reference + collision only

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_poke_prototype.blend")


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_collection in (bpy.data.meshes, bpy.data.materials):
        for block in list(block_collection):
            if block.users == 0:
                block_collection.remove(block)


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def build_ground():
    bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(make_material("GroundMat", (0.75, 0.75, 0.75)))

    select_only(ground)
    bpy.ops.object.modifier_add(type="COLLISION")
    # (Blender 5.1 verified: the old standalone `bpy.ops.object.collision_add()`
    # operator no longer exists -- Collision physics is added via the unified
    # modifier system, same as Soft Body. `obj.collision` is populated as soon
    # as the "Collision" modifier is added.)
    # Defaults are fine for a static ground plane -- no override needed.
    return ground


def build_blob():
    bpy.ops.mesh.primitive_ico_sphere_add(radius=BLOB_RADIUS, subdivisions=BLOB_SUBDIV,
                                           location=(0, 0, 0))
    blob = bpy.context.active_object
    blob.name = "Blob"

    # Squash (non-uniform scale), applied immediately so the Soft Body solver's
    # rest shape is the squashed shape, not the original sphere.
    blob.scale = (1.0, 1.0, BLOB_SQUASH_Z)
    select_only(blob)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Organic per-vertex noise displacement along vertex normals, so this reads
    # as a blob rather than a perfect (squashed) sphere.
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(blob.data)
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        n = noise.noise(v.co * BLOB_NOISE_SCALE)
        v.co += v.normal * (n * BLOB_NOISE_AMPLITUDE)

    bm.normal_update()

    # The noise displacement above moves each vertex along its own normal with
    # no floor awareness -- on an unlucky seed that can leave a real gap above
    # the ground (unwanted pre-poke falling motion) or push vertices through it
    # (interpenetration), depending on which way the noise happens to point at
    # the bottom pole. Rather than distorting the organic shape to compensate,
    # measure the ACTUAL lowest vertex post-noise and place the whole object so
    # that point sits at exactly world z=0 -- this guarantees zero-gap,
    # zero-penetration ground contact regardless of the noise seed, without
    # touching the noise pattern itself.
    zs = [v.co.z for v in bm.verts]
    z_min, z_max = min(zs), max(zs)
    base_offset = -z_min  # object location.z that puts the lowest vertex at world 0

    # Determine the base ring/cap to pin (bottom BASE_PIN_HEIGHT_FRACTION of the
    # blob's actual post-deformation height), from these same real, final
    # vertex Zs -- not a hardcoded absolute value, so this stays correct if the
    # geometry constants above ever change.
    pin_threshold_local = z_min + BASE_PIN_HEIGHT_FRACTION * (z_max - z_min)
    pinned_indices = [v.index for v in bm.verts if v.co.z <= pin_threshold_local]
    rest_indices = [v.index for v in bm.verts if v.co.z > pin_threshold_local]

    bm.to_mesh(blob.data)
    bm.free()
    blob.data.update()

    blob.location = (0, 0, base_offset)
    bpy.ops.object.shade_smooth()
    blob.data.materials.append(make_material("BlobMat", (0.85, 0.55, 0.55)))

    # Root cause of the base drifting/sliding instead of staying put: with a
    # single flat goal_default applied to every vertex (including the base) and
    # no vertex-group pinning, nothing anchors the blob to the ground -- the
    # whole connected spring mesh is free to sag/slide together under gravity,
    # not just deform locally where poked. Fix: a vertex group covering EVERY
    # vertex (not just the base) with two weight levels -- base ring/cap at
    # ~1.0 (effectively pinned) and everything else at SB_GOAL_DEFAULT. This
    # has to cover every vertex, not just the base: once a vertex_group_goal is
    # assigned, Blender computes each vertex's effective goal from its weight
    # in that group, and a vertex left out of the group gets implicit weight 0
    # -- it does NOT fall back to goal_default. Leaving the rest of the mesh
    # unassigned would have silently removed their goal constraint entirely.
    vg = blob.vertex_groups.new(name="BasePin")
    vg.add(pinned_indices, BASE_PIN_GOAL_WEIGHT, "REPLACE")
    vg.add(rest_indices, SB_GOAL_DEFAULT, "REPLACE")

    select_only(blob)
    bpy.ops.object.modifier_add(type="SOFT_BODY")
    sb = blob.modifiers["Softbody"].settings
    sb.mass = SB_MASS
    sb.goal_default = SB_GOAL_DEFAULT  # superseded by the vertex group below for
                                        # every vertex, kept in sync as a fallback
    sb.damping = SB_DAMPING
    sb.vertex_group_goal = "BasePin"
    # Everything else (friction, goal_spring, goal_friction, goal_min, goal_max,
    # pull, push, bend, use_self_collision, ...) is left at Blender's verified
    # default -- see the constants block above for the exact values. Note
    # goal_min=0.0/goal_max=1.0 (both defaults, unchanged) means the vertex
    # group weight IS the effective goal value directly (no min/max remapping).

    print("\n--- Soft Body settings actually written to the .blend (Blob object) ---")
    for f in ("mass", "friction", "use_goal", "goal_default", "goal_spring", "goal_friction",
              "goal_min", "goal_max", "vertex_group_goal",
              "use_edges", "pull", "push", "bend", "damping", "use_self_collision"):
        print(f"  {f} = {getattr(sb, f)}")
    print(f"BasePin vertex group: {len(pinned_indices)} vertices @ weight {BASE_PIN_GOAL_WEIGHT} "
          f"(base), {len(rest_indices)} vertices @ weight {SB_GOAL_DEFAULT} (rest)")
    print(f"Ground contact: actual post-noise local z range [{z_min:.5f}, {z_max:.5f}] "
          f"(height {z_max - z_min:.5f} m vs. nominal {BLOB_HEIGHT:.5f} m), object placed at "
          f"z={base_offset:.5f} so the lowest vertex lands exactly at world z=0")

    return blob


def build_probe():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=PROBE_RADIUS,
                                          location=(PROBE_XY[0], PROBE_XY[1], PROBE_START_Z))
    probe = bpy.context.active_object
    probe.name = "Probe"
    bpy.ops.object.shade_smooth()
    probe.data.materials.append(make_material("ProbeMat", (0.75, 0.15, 0.15)))

    select_only(probe)
    bpy.ops.object.modifier_add(type="COLLISION")
    # (Blender 5.1 verified: the old standalone `bpy.ops.object.collision_add()`
    # operator no longer exists -- Collision physics is added via the unified
    # modifier system, same as Soft Body. `obj.collision` is populated as soon
    # as the "Collision" modifier is added.)
    # Defaults are fine here too -- the probe only needs to physically displace
    # the blob's Soft Body mesh on contact, which Collision physics handles
    # regardless of the object being keyframed (not itself Rigid Body/Soft Body).

    keyframes = [
        (FRAME_START, PROBE_START_Z),
        (FRAME_DESCENT_END, PROBE_INDENT_Z),
        (FRAME_HOLD_END, PROBE_INDENT_Z),
        (FRAME_END, PROBE_START_Z),
    ]
    for frame, z in keyframes:
        bpy.context.scene.frame_set(frame)
        probe.location = (PROBE_XY[0], PROBE_XY[1], z)
        probe.keyframe_insert(data_path="location", frame=frame)

    # "Steady rate" descent/hold/withdrawal, per the brief -- Blender's default
    # Bezier (ease-in/ease-out) interpolation would NOT be a steady rate, so
    # force linear interpolation on every inserted keyframe.
    # Blender 5.1 verified: Action fcurves live under the new layered-animation
    # model (action.layers[0].strips[0].channelbags[0].fcurves), not the old
    # flat action.fcurves -- fall back to the old path if it's ever available.
    action = probe.animation_data.action
    if hasattr(action, "fcurves") and len(action.fcurves) > 0:
        fcurves = action.fcurves
    else:
        fcurves = action.layers[0].strips[0].channelbags[0].fcurves
    for fcurve in fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = "LINEAR"

    bpy.context.scene.frame_set(FRAME_START)
    return probe


def build_camera_and_light():
    """Minimal scaffolding so the file isn't empty on open -- not physics-relevant,
    feel free to replace/reposition once you're set up in the GUI."""
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (2.6, -2.6, 1.8)
    cam.rotation_euler = (1.15, 0, 0.785)
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("Light", type="SUN")
    light_data.energy = 2.5
    light = bpy.data.objects.new("Light", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (2, -2, 4)
    light.rotation_euler = (0.6, 0.2, 0.8)


def main():
    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.render.fps = FPS

    clear_scene()
    build_ground()
    blob = build_blob()
    build_probe()
    build_camera_and_light()

    bpy.ops.wm.save_as_mainfile(filepath=OUT_PATH)

    print("\n" + "=" * 78)
    print("PROBE POKE PROTOTYPE -- key numbers")
    print("=" * 78)
    print(f"Blob: radius={BLOB_RADIUS} m, subdivisions={BLOB_SUBDIV}, squash_z={BLOB_SQUASH_Z}, "
          f"noise_amplitude={BLOB_NOISE_AMPLITUDE} m (scale={BLOB_NOISE_SCALE})")
    print(f"Blob nominal resting height: {BLOB_HEIGHT:.4f} m (actual post-noise height printed "
          f"above by build_blob(); object placed at z={blob.location.z:.5f} so its true lowest "
          f"vertex sits exactly at world z=0, regardless of the noise seed)")
    print(f"Probe: radius={PROBE_RADIUS} m, start_z={PROBE_START_Z:.4f}, "
          f"indent_z={PROBE_INDENT_Z:.4f} (indent depth={PROBE_INDENT_DEPTH:.4f} m, "
          f"{PROBE_INDENT_FRACTION*100:.0f}% of blob height)")
    print(f"Frames: start={FRAME_START} descent_end={FRAME_DESCENT_END} "
          f"hold_end={FRAME_HOLD_END} end={FRAME_END}  @ {FPS} fps "
          f"({FRAME_END/FPS:.2f} s total)")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
