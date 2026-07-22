# Probe poke prototype — Blender Soft Body (NOT the real MPM solver)

Fast visual mockup of the rigid-probe-poke interaction (sim 006 concept), built with
**Blender's built-in Soft Body (mass-spring) physics** — a completely different
numerical model from the Taichi MLS-MPM material point method used for the real
dataset (sims 001-005). There are no material points, no grid, no constitutive
stress law, no Lamé/yield/friction parameters here — just mesh edges modeled as
springs, plus an optional "goal" term pulling vertices back toward their rest
position.

**This is a geometry/motion/visual-feel sanity check only — not a physics
reference.** Never compare its deformation magnitude, timing, or anything
quantitative against the real `GtX`/`GtF`/`GtStress` tensors or
`dataset_viewer.html` entries for sims 001-005.

Completely self-contained: no shared files, imports, or config with
`simulation/`, `dataset_viewer.html`, or `experiments/proxy_net/`.

## Regenerate the scene

```bash
blender --background --python build_scene.py
```

Procedurally rebuilds and overwrites `probe_poke_prototype.blend`. Physics is left
**unbaked** and nothing is rendered — open the file in the Blender GUI, tweak Soft
Body settings on the `Blob` object if needed, bake (Physics tab → Soft Body →
Cache → Bake), and scrub/render interactively from there.

Built and verified against **Blender 5.1.2** (snap install, same as the real
render pipeline). Two Blender API surfaces changed since older tutorials/docs you
may find online, both handled in `build_scene.py`:
- `bpy.ops.object.collision_add()` no longer exists — Collision physics is added
  via the unified modifier system: `bpy.ops.object.modifier_add(type="COLLISION")`.
- `Action.fcurves` is no longer a flat list — fcurves now live under Blender's
  layered-animation model:
  `action.layers[0].strips[0].channelbags[0].fcurves`.

## What's in the scene

| Object | Role | Physics |
|---|---|---|
| `Ground` | flat plane, visual reference | Collision (passive, defaults) |
| `Blob` | subdivided icosphere, squashed + Perlin-noise-displaced so it reads as an organic blob, not a perfect sphere | Soft Body (mass-spring) |
| `Probe` | rigid sphere, keyframed straight down and back up | Collision (kinematic — not itself Soft Body) |
| `Camera` / `Light` | minimal scaffolding so the file isn't empty on open | none — reposition freely |

## Exact values chosen (all defined as named constants at the top of `build_scene.py`)

**Blob geometry:**
- Icosphere radius 0.5 m, subdivision level 3 (162 vertices) — enough smoothness
  for an interactive Soft Body sim without being heavy.
- Squashed by a non-uniform scale of 0.85 on Z before adding physics (so the
  Soft Body rest shape is already squashed, not a perfect sphere settling into
  one).
- Organic look: each vertex displaced along its own normal by Perlin noise
  (`mathutils.noise.noise`), amplitude ±0.035 m (~7% of radius), spatial
  frequency scale 2.5 — gentle, not shape-distorting.
- Nominal resting height: 0.85 m. The object's actual placement is computed
  from its real post-noise geometry (lowest vertex measured and placed exactly
  at world z=0), not this nominal number, so ground contact is exact regardless
  of the noise seed — see the "base drift" fix below.

**Base anchoring (fixed — was drifting/sliding):** the blob used to have zero
vertex pinning, so a single flat `goal_default` applied to every vertex
(including the base) left nothing anchoring it to the ground — the whole
connected spring mesh was free to sag/slide together under gravity instead of
staying put while only the poked region deformed. Root cause confirmed by
direct inspection before fixing (no keyframes on the blob itself, so it wasn't
that; `sb.vertex_group_goal` was empty and `Blob.vertex_groups` was `[]` — zero
pinning existed). Fixed with a `BasePin` vertex group covering **every**
vertex — bottom 15% of the blob's actual height at weight 0.98 (effectively
anchored), everything else at 0.3 (same as the old flat `goal_default`, so the
poked region's deformability is unchanged). Every vertex needed an explicit
weight here, not just the base ring: once `vertex_group_goal` is assigned,
Blender computes each vertex's effective goal from its weight in that group,
and a vertex left out of the group gets implicit weight 0 — it does **not**
fall back to `goal_default`. Verified after rebuilding: ground contact gap is
now exactly 0.0 (was ~0.27 mm, working only by luck of the noise seed before),
and the vertex group covers all 162 vertices (26 base / 136 rest).

**Soft Body settings** (Blender 5.1.2 defaults verified via a throwaway probe
script before writing the real one — see `build_scene.py`'s header comment for
the full verified-default list). Only two values deliberately changed from
Blender's defaults:
- `goal_default`: **0.7 → 0.3** — the default pulls too strongly back toward the
  rest shape and visibly resists the probe's indentation; 0.3 still prevents
  total collapse but lets the poke actually show.
- `damping`: **0.5 → 5.0** — with an animated collision probe pushing hard into
  the mesh, the default under-damps and risks sustained jitter/instability;
  5.0 is a safer starting point per the "doesn't explode" requirement. Lower it
  back down for more jelly-like wobble once you're baking interactively.
- Everything else (`mass=1.0`, `friction=0.5`, `pull=push=0.5`, `bend=0.0`,
  `use_self_collision=False`, ...) is left at Blender's verified default.
- `vertex_group_goal = "BasePin"` (see "Base anchoring" above) — this is what
  keeps the base anchored while the rest of the mesh deforms freely.

**Probe:**
- Radius 0.15 m (30% of blob radius).
- Starts 0.3 m clear above the blob's resting top surface (center z=1.30).
- Descends to indent 25% of blob height (0.2125 m) into the blob (center
  z=0.7875) — inside the requested 20-30% range.
- Collision physics only (not Rigid Body, not Soft Body) — a kinematic
  (keyframed) Collision object is exactly what Blender needs to let it displace
  a Soft Body mesh on contact.

**Timing** (24 fps, matching the real render pipeline's convention):
- Frame 1: start (probe clear above the blob).
- Frames 1→40 (~1.6 s): steady linear descent to full indentation.
- Frames 40→60 (~0.8 s): held at full indentation.
- Frames 60→100 (~1.7 s): steady linear withdrawal back to start.
- Total: 100 frames, ~4.17 s.
- All probe keyframes forced to **LINEAR** interpolation (not Blender's default
  Bezier ease-in/out) so the descent/hold/withdrawal is a genuinely steady rate,
  per the brief.

## Verification performed (no bake, no render)

Reopened the saved `.blend` fresh and checked: frame range/fps, all 5 objects
present, `Ground`/`Probe` have a `Collision` modifier, `Blob` has a `Softbody`
modifier with the exact settings above, `Blob` vertex count and world position,
`Probe` radius, and all 4 keyframe frame/value/interpolation triples on the
Z location fcurve. Everything matched the intended values exactly.

Re-verified after the base-anchoring fix: `Blob.animation_data` is still `None`
(zero keyframes on the blob itself), ground contact gap is exactly `0.0`,
`BasePin` vertex group covers all 162 vertices (26 @ weight 0.98, 136 @ weight
0.3), `sb.vertex_group_goal == "BasePin"`, gravity confirmed enabled and
reasonable (`(0, 0, -9.81)`), and all 4 probe keyframes are unchanged. Still no
baking, no rendering.
