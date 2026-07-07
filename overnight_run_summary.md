# Overnight Run Summary

Status: **COMPLETE** (resumed 2026-07-07 ~12:18 CDT, finished ~14:50 CDT — autonomous run per user
instruction, no check-ins during execution)

## Plan (7 steps, resumed mid-run after a session restart)

Confirmed done before this resume (per user): 001 elastic fully rendered (40/40 frames, GIF
verified); 002 plasticine sim+keyframes+diagnostics+E/nu+gitignore done; 003/004/005 UniPhy sims
completed with verified output tensors.

All 7 remaining steps completed:
1. Keyframes for 003/004/005
2. Diagnostics + effective material metrics for 003/004/005
3. `.gitignore` entries for 003/004/005 large tensors
4. Blender pipeline render for 002/003/004/005 → 40-frame PNG + GIF each
5. Add all 5 sims to `dataset_viewer.html`
6. Commits (separate logical commits) + push to `origin/main`
7. This summary file

---

## Completed

- **Step 1 — keyframes**: `sphere_keyframes.npy` computed for 003 (newtonian), 004 (sand), 005
  (non_newtonian) via the same method as `compute_keyframes.py` (40 samples linspace over first
  2000 substeps, center/extent from `GtX.pt`). All saved under each sim's `output/` dir.

- **Step 2 — diagnostics plots**: `shape_diagnostics_00{3,4,5}.png` generated in
  `simulation/renders/` (bounding-box extents + deformation magnitude vs. frame), same template
  as 001/002.

- **Step 2 — effective material metrics** (regressed directly from `GtF.pt`/`GtStress.pt`/`GtC.pt`,
  same rigorous stress/strain-fit method as 002's plasticine E/nu measurement):

  - **005 non_newtonian** (`material_type: von_mises`, same constitutive model as plasticine, just
    different mu/lam/yield_stress/plastic_viscosity): pre-yield Hencky stress/strain regression
    (156597 pre-yield samples, mask `dev_strain_norm < 0.35`, safely below the analytic yield onset
    ~0.408 for this config's yield_stress=1000/mu=1000) gives **E_eff = 2499.9993, nu_eff = 0.2500**
    vs. nominal E_in=2500.0, nu_in=0.25 from config mu=lam=1000. R²=1.0 on both trace and deviatoric
    fits — clean fit, E/nu is fully meaningful here since it's the same von Mises elastoplastic
    model as plasticine.

  - **004 sand** (`material_type: drucker_prager`): the constitutive stress formula is the *same*
    Hencky-strain-based formula as von_mises/elasticity (only the plastic return-mapping differs),
    so a small-strain/pre-yield E/nu fit is equally meaningful in the elastic regime before
    frictional yielding kicks in. friction_alpha=0.01 is tiny, so this system yields readily under
    confinement, but a `dev_strain_norm < 0.02` (and up to 0.15, checked several thresholds — result
    identical, ~107203-138772 samples depending on threshold) mask isolates the pre-yield elastic
    response: **E_eff = 5999.998, nu_eff = 0.2500** vs. nominal E_in=6000.0, nu_in=0.25 (config
    mu=lam=2400). R²=1.0. This E/nu characterizes only the small-strain elastic response; sand's
    actual large-deformation/flow behavior is governed by friction_alpha (frictional yield surface
    slope), which E/nu does not capture — flagged in both this log and the dataset viewer rather
    than silently omitted. Interestingly, final per-particle deformation stretch is ~0 even though
    the pile is heavily spread on the floor — granular particles rearrange plastically (individual
    F stays near-identity) rather than each particle stretching, unlike the elastic/plasticine
    spheres. Noted in the viewer so it doesn't read as a bug.

  - **003 newtonian** (`material_type: viscous_fluid`): **does NOT support an E/nu fit** — confirmed
    by reading the stress kernel in `mpm_simulator_perparticleparams.py`: for this material only
    `F[...,0,0]` is tracked (a scalar volume-ratio proxy, not a real deformation-gradient tensor),
    and the stress law is `stress = kappa*(J-1/J^6)*I + mu*J*sym(C)` — i.e. stress depends on the
    strain *rate* (via `C`, the affine velocity field), not on accumulated strain, so there is no
    elastic restoring modulus to measure. Fit the physically-appropriate equivalent instead —
    effective shear viscosity and bulk-viscosity term, regressed the same way (linear fit through
    origin, deviatoric and volumetric stress components against the kinematic quantities the
    constitutive law says they're proportional to), using `GtC.pt` (the affine velocity/rate-of-
    deformation field, needed here since `GtF.pt` alone doesn't carry shear information for this
    material): **mu_eff = 49.657** (shear viscosity, R²=0.999) vs. nominal mu_in=50.0, and
    **kappa_eff = 63.399** (bulk viscosity term, R²=0.9999) vs. nominal kappa_in = (2/3)*50+30 =
    63.333. Config's `mu`/`lam` fields are being reused as viscosity coefficients for this material
    (not Lame elastic constants) — noted so this isn't confused with the elastic sims' mu/lam.

- **Step 3 — .gitignore**: added `output/00{3,4,5}_..._sphere_free_fall_test/Gt{C,F,Ftmp,Stress}.pt`
  entries to `simulation/.gitignore`, following the exact 001/002 pattern, plus a new entry for
  Hydra's own `data_process/outputs/` run-log directory (config snapshot + log per run, not
  simulation data — was accumulating untracked clutter from each `dataset_generate.py` invocation).
  Verified all paths with `git check-ignore -v` — all correctly ignored.

- **Step 4 — Blender rendering**: headless render (7 scripts: 01,02,03,04,05,06,09, ~50s/frame at
  384 Cycles/OptiX samples) for all of 002/003/004/005, sequentially on the single GPU. Each sim
  rendered to its own `simulation/renders/frame_00X/` (001's existing `frame_/` untouched), then
  encoded to `sphere_drop_00X.gif` via the same ffmpeg palettegen/paletteuse pipeline documented in
  `blender/scripts/README.md`. Total wall time ~2h35m. All 4 confirmed: exit code 0 for both the
  Blender render and the gif encode, exactly 40 PNG frames each, gif sizes 8.6-8.9 MB (in line with
  001's 8.9 MB). Intermediate `palette.png` ffmpeg artifacts removed from each frame dir before
  committing (001's `frame_/` didn't have one either).

- **Step 5 — dataset_viewer.html**: added sidebar entries + full `SIMULATIONS` object entries for
  002/003/004/005, mirroring 001's structure/fields exactly (framework/scene/object/material/
  results/raw_metadata/files), populated with the real keyframe/diagnostics/validation/E-nu-or-
  viscosity numbers computed above, plus the real GIF file sizes once rendering finished. Two
  small, targeted template edits were needed (not present in 001, which doesn't need them):
  (a) the E/Young's-modulus and nu/Poisson's-ratio cards now render "N/A" with an explanatory
  sub-caption when `material.E`/`material.nu` are `null` (needed for 003 newtonian, a rate-dependent
  fluid with no elastic modulus) instead of crashing on `null.toLocaleString()`; (b) the mu/lambda
  card labels are now overridable per-sim via optional `mu_label`/`lam_label`/`mu_unit`/`lam_unit`
  fields (003 labels them as shear/bulk viscosity coefficients in Pa·s, not Lamé parameters); (c) the
  "final extents" caption's flattening percentage is now computed from the actual data instead of a
  hardcoded "~45%" (which was 001-specific). Verified with a string-aware brace/bracket matcher
  (plain bracket counting isn't reliable through JS string literals) that the file is structurally
  balanced, and confirmed all 8 referenced image files (4 gifs + 4 diagnostics pngs) exist on disk
  before committing. Also opened the file in Firefox as a spot check (no screenshot tool available
  in this environment to capture/verify pixel output programmatically, so this was launch-only).

- **Step 6 — commits + push**: 5 commits total, each scoped to one logical unit (verified with
  `git status`/`git diff --cached` before each so no unrelated files leaked across boundaries —
  caught and fixed one slip early on where 003/004/005 files briefly got swept into the first
  commit, corrected via `git reset --soft` since nothing had been pushed yet):
  1. `Finish 001 elastic sphere Blender re-render: camera/room/lighting fixes`
  2. `Add plasticine (002) sphere free-fall sim: keyframes, diagnostics, E/nu fit`
  3. `Add newtonian/sand/non-Newtonian (003-005) sphere free-fall sims`
  4. `Render plasticine/newtonian/sand/non-Newtonian (002-005) through Blender`
  5. `Add all 5 sims to dataset_viewer.html`

  Pushed to `origin/main` after all 5 commits landed (see below for confirmation).

## Failed / needs attention

*(none — every step completed cleanly, all Blender renders exited 0 with the expected 40 frames)*

## Judgment calls made

- The original step-4 instruction said "run each of the **3 new sims** through the Blender
  pipeline," but step 5 asked to add **all 5 sims** to the dataset viewer following the elastic
  entry's structure — which includes a `video` (GIF) field. 002 plasticine had never actually been
  Blender-rendered (only `shape_diagnostics_002.png` existed, no GIF/frame sequence), so rendered
  002 as well to keep the viewer consistent. Flagging this interpretation in case it wasn't
  intended — reverting is just deleting `simulation/renders/frame_002/` + `sphere_drop_002.gif` and
  the corresponding viewer entry's video/diagnostics fields if so.
- "All 9 scripts" in the original instruction refers to the scripts' numeric naming (last one is
  `09_render_settings.py`), not a literal file count — `07_wall_decoration.py` and
  `08_surgical_instruments.py` were deleted earlier this session, so the actual pipeline run is the
  7 remaining scripts (01,02,03,04,05,06,09), matching what was already used to produce the
  verified 001 render.
- For sand/non_newtonian's E/nu regression, used the exact same pre-yield-mask + linear-regression-
  through-origin method as plasticine's existing script rather than inventing a new one per
  material, since the underlying stress kernel formula is identical for all three (von_mises,
  drucker_prager materials share it) — only the yield-mask threshold was adapted per material's
  yield_stress/friction_alpha.
- For newtonian (003), added `mu_label`/`lam_label`/`mu_unit`/`lam_unit` and a `material.E: null` /
  `material.nu: null` convention to `dataset_viewer.html`'s data model, plus the two small template
  guards described in step 5 above, rather than fabricating a fake E/nu number just to fit the
  existing template shape. This is a data-model extension, not a redesign — 001/002/004/005 are
  unaffected (their `material.E`/`nu` stay plain numbers, same as before).

## Worth a look in the morning

- Please sanity-check the two `dataset_viewer.html` template edits (E/nu → "N/A" guard, and the
  mu/lambda label override) render the way you'd want — I didn't have a way to screenshot the
  actual browser output in this environment, only a structural (brace-balance) check and confirming
  every referenced image file exists on disk.
- The judgment call above about rendering 002 (not just 003/004/005) through Blender — flagged in
  case the omission from the original step-4 wording was intentional rather than an oversight.
