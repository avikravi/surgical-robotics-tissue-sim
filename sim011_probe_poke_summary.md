# Sim 011 — Rigid Probe Poking Elastic Tissue: Run Summary

Status: **COMPLETE** — working GIF rendered, spot-checked, and committed by
12:58pm CDT, ahead of the 1pm target (see timeline note at bottom for why the
mid-run estimate was pessimistic).

## Goal

Sim 011: a rigid probe poking an elastic tissue sphere, time-boxed to a working
demo by 1pm (2026-07-15). 011, not 006 — 006-010 are reserved for a separate
task (re-running materials 001-005 through the neural-network path).

## Step 0 — investigation (before writing anything)

Checked `simulation/data_process/mpm_simulator_perparticleparams.py`'s existing
collider mechanism (`add_surface_collider` for the floor, `add_cylinder_collider`)
against `sdf_functions.py` (unrelated — that module is for sampling the initial
particle cloud from a shape, not runtime collision).

Finding: `grid_op` already generically applies every closure in
`self.analytic_collision`, so the mechanism itself generalizes cleanly to a
second collider — no changes needed there. But both existing colliders bake
their position in as a **compile-time Python constant** captured by the Taichi
closure (`ti.Vector(point)` from a plain Python list), which cannot move
between substeps. Reported this back before proceeding, per the brief, and
proposed the smallest change: a new `add_sphere_collider()` using the same
three-branch sticky/slip/separate logic, but taking `center` as a
`ti.Vector.field(dim, dtype, shape=())` that the caller writes to once per
substep (a plain field write, not a kernel) instead of a baked constant.

## Step 1 — implementation (kept minimal, per the brief)

- **Material**: 001a (soft elastic, E≈15kPa) verbatim, not 001b. Reasoning:
  001a has the most CFL headroom (~12x margin, vs. ~2-10x for 001b) of the
  validated elastic presets — the safest choice for absorbing whatever extra
  numerical stress a brand-new, never-before-tested collider interaction adds,
  which mattered given the time budget didn't allow for an
  instability-debugging loop.
- **Probe collider**: new `add_sphere_collider()` in
  `mpm_simulator_perparticleparams.py`, added right next to the existing
  colliders, same pattern.
- **Driver script**: new `simulation/data_process/dataset_generate_011_probe_poke.py`
  (does not touch `dataset_generate.py`). Reuses `MPMWrapper` completely
  unmodified for particle init, material setup, and the floor collider (which
  already picks `surface_separate` for `elasticity`, per the earlier
  per-material floor-collider fix — free lunch, no new code needed there).
  Reuses the already-fixed dynamic `buffer_len` sizing (`num_sim_steps *
  substep_gt + 2`), not the old fixed-4096 bug.
- **Probe trajectory** (hardcoded, keyed on substep index, no config
  plumbing): probe stays parked at `y=1.0` (out of the way) for substeps
  0-1400 — the sphere free-falls and (it turned out) is *still falling* at
  substep 1400, not yet resting, so the probe's descent and the floor impact
  end up overlapping rather than being cleanly sequential (see "judgment
  calls" below). Descends to `y=0.02` over substeps 1400-1700, holds
  1700-1850, withdraws back to `y=1.0` over 1850-2000. Probe radius 0.06,
  fixed at the sphere's (x, z) = (0.55, 0). `y=0.02` was picked to *guarantee*
  contact regardless of exact settle height, not to hit a tuned indentation
  percentage — Step 2 only required confirming visible deformation happens.
- One-way coupling only, as specified: the probe imposes a boundary condition
  on nearby grid nodes via `surface_separate`; it is not itself pushed by the
  tissue (no physics on the probe side).
- Output: `simulation/output/011_elastic_sphere_probe_poke_test/` with the
  same `GtX`/`GtV`/`GtC`/`GtF`/`GtFtmp`/`GtStress` convention as every other
  sim, plus `probe_keyframes.npy` (40-sample downsampled probe trajectory —
  single source of truth for the Blender render step, so the render can't
  drift from the actual simulated trajectory). `GtC`/`GtF`/`GtFtmp`/`GtStress`
  gitignored the same way as every other sim (71MB each).

## Step 2 — validation (before touching Blender)

Ran in 17.6s, 2000 substeps, **no CFL violation**. Checked directly (not just
via the diagnostics plot):
- `GtX`/`GtF`: no NaN, no Inf.
- Deformation-gradient singular values bounded 0.43-1.67 (no blow-up).
- **Dramatic, unambiguous contact**: vertical extent drops from 0.199 (undeformed)
  to as low as 0.052 sim units during the hold phase (~74% reduction) — far
  more than "some visible deformation." Material bulges outward in x/z as
  expected for a squashed near-incompressible elastic solid. Partial spring-back
  visible during withdrawal (extent climbs back to ~0.072 by the end, not fully
  recovering — physically reasonable given the compound floor+probe impact).
- See `simulation/renders/shape_diagnostics_011_elastic_sphere_probe_poke_test.png`.

This cleared the "stop and report if unstable/no contact" bar, so proceeded to
Blender without pausing.

## Step 3 — Blender render

Reused the existing headless pipeline (env-var-driven `SIM_OUTPUT_DIR`), with
one addition: `simulation/blender/scripts/07_animate_probe.py` (new,
run between `06_camera.py` and `09_render_settings.py`). This script is
**additive-only and self-skipping** — it checks for `probe_keyframes.npy` in
the sim's output dir and does nothing if it's absent, so it's safe to leave in
the shared `scripts/` folder without affecting a re-render of any of the other
10 sims. Probe rendered as a metallic UV sphere (radius 0.06, matching the sim
exactly), keyframed with the same sim-to-Blender coordinate remap as the
tissue sphere.

Encoded via the same ffmpeg palettegen/paletteuse pipeline as every other sim,
to `simulation/renders/sphere_drop_011_elastic_sphere_probe_poke_test.gif`
(9.4MB, in line with the other sims' 8-9MB GIFs).

Spot-checked two frames directly: frame 10 shows the tissue sphere still
falling, undeformed, probe not yet in shot (parked at y=1.0, out of frame —
matches the "parked" phase); frame 34 (hold phase) shows the metallic probe
sphere visibly seated in the dramatically flattened pink tissue disc — a
clear, unambiguous poke shot.

## Judgment calls / things worth a look

1. **The probe's descent overlaps the floor impact rather than following it.**
   I assumed (based on other sims' floor-contact timing) the sphere would
   already be resting by substep 1400; it's actually still in free-fall then.
   The result is still a clean, dramatic, stable demo (see Step 2), but it's a
   compound "falling + poked" impact rather than a "land, settle, then get
   poked" sequence. If you want the cleaner sequential version, the fix is
   just pushing `SUBSTEP_PARK_END` later (e.g. ~1700+) and shortening the
   poke window accordingly, then re-validating — did not do this given the
   time budget, since the current result already passed Step 2's bar.
2. **`y=0.02` poke depth is a "guaranteed contact" choice, not a tuned
   indentation percentage** — unlike the earlier Blender-only prototype
   (`blender_prototypes/probe_poke_prototype/`), which targeted a specific
   20-30% indentation. Worth tuning if the compound impact in point 1 above
   ends up looking too extreme once you see the GIF.
3. Did not run `analyze_sim.py`'s full free-fall/floor-contact validation
   metrics (used for the `dataset_viewer.html` entries) — only the NaN/Inf/
   singular-value/extent checks the brief specifically asked for in Step 2.
   Worth running if this sim is going to get a `dataset_viewer.html` entry
   later (explicitly out of scope for today, per the brief).

## Step 4 — commits

Three commits, each checked with `git status`/`git diff --cached --stat`
first so nothing unrelated leaked in (in particular, `blender_prototypes/` —
untracked leftover from an earlier, unrelated session — was deliberately never
staged):
1. `Add moving sphere collider + sim 011 driver: rigid probe poking elastic tissue`
   — the two new/changed code files.
2. `Add sim 011 output data: elastic sphere + probe poke (GtX/GtV, keyframes, diagnostics)`
   — `.gitignore` + small tracked output files.
3. `Render sim 011 through Blender: rigid probe poking elastic tissue sphere`
   — 40 PNG frames + the final GIF.

Per the brief, `dataset_viewer.html`'s 5 material entries and pair-grid layout
were not touched, and no 011 entry was added there — will ask before doing
that if there's time left over.

## Timeline note

Flagged early (12:25pm, when this task started with 35 minutes on the clock)
that the full Blender-rendered GIF would likely land a few minutes past 1pm,
based on ~45-60s/frame observed both in an earlier session's render batch and
in the first few frames of this run. That per-frame rate held for most of the
run, but the actual finish (12:58pm) came in under the extrapolated estimate —
the early-frame samples I estimated from were apparently on the slower end
(camera/scene setup and Cycles denoiser warm-up cost is front-loaded into the
first few frames, so extrapolating straight from frame 3-4's pace overshot the
true average). Net: sim data (Steps 0-2) validated and committed by ~12:34pm;
render started ~12:31pm, finished and GIF-encoded by 12:58pm — under the 1pm
target after all, despite the mid-run estimate suggesting otherwise.
