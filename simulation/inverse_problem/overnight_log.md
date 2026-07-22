# Overnight task log — 001a calibration script + viewer panel

## TOP SUMMARY (read this first)

**Everything completed and pushed to `origin/main`. No stop conditions were hit, nothing
is sitting as an uncommitted local diff.**

What got done, in 4 commits on top of `778dfd7` (the state before this task started):

1. `044f302` — amended (was already committed locally as `dee3f4e` from earlier in the
   session, before this task list was given; un-pushed, so amended rather than
   double-committing) to also gate the per-call timing-breakdown print behind a new
   `--verbose` flag, so default output is just the `E=... nu=... loss=...` line.
2. `29dcbf8` — added per-evaluation trajectory logging (`trajectory_summaries` +
   `ground_truth_trajectory` in the output JSON) and regenerated `001a_calibration.json`.
3. `0e4ae3d` — added the additive "Inverse Problem" viewer panel to `dataset_viewer.html`
   (parameter-space search path, trajectory overlay, slider, stats strip). Validated
   (Step 5) before committing: JSON valid, brace/bracket/backtick-balanced, headless-
   Firefox screenshots of both the untouched default view and the new panel confirm real
   rendering with real data, diff is purely additive (384 insertions, 0 deletions) to the
   existing 10-sim viewer.
4. `2d611d6` — documented the calibration script and viewer panel in CLAUDE.md/README.md.

**Result on 001a**: recovers E=14894.86 (true 14998.84, 0.69% off), ν=0.4486 (true
0.4500, 0.30% off) over 28 evaluations. A diagnostic sweep (Step-1-equivalent from the
prior conversation turn, re-confirmed implicitly by this run's clean convergence) shows
ν is strongly physically identifiable once floor contact is in the optimization window --
this was a real, working result, not a fluke.

**Scope check**: confirmed via `git diff --stat 778dfd7..HEAD` that only 5 files changed
across all 4 commits (`CLAUDE.md`, `README.md`, `dataset_viewer.html`,
`simulation/inverse_problem/calibrate_material.py`,
`simulation/inverse_problem/results/001a_calibration.json`) -- nothing under
`simulation/output/` touched, none of the other 9 sims touched.

**Deviations from the task's literal instructions, all flagged in the step-by-step log
below with reasoning** (nothing silently changed): the task assumed an existing
fetch()-based data-loading convention in `dataset_viewer.html` to match -- there wasn't
one (the existing viewer's data is hand-authored inline, not fetched); used fetch()
anyway since that's what the task explicitly asked for for this panel. The task's design
spec said "Arial" -- used the page's actual existing font (JetBrains Mono) instead, since
Arial doesn't appear anywhere else on the page. The task described "last" and "best"
evaluation as synonymous for the slider's default position -- they're not quite identical
in the actual logged data, so the slider defaults to the true argmin(loss) entry instead
of the literal last array index.

**What to check first when you're back:**
1. Open `dataset_viewer.html` (locally or via the GitHub Pages link) and click
   "001a · Calibration" in the sidebar -- confirm the panel looks right to your eye (I
   only validated via headless screenshot + spot-checked values, not interactively).
2. Try dragging the slider -- I verified the underlying render function works (same code
   path as the initial render, which did work), but didn't screenshot the drag
   interaction itself frame-by-frame.
3. Skim the CLAUDE.md/README.md diffs (in Step 6's log entry below) for tone/accuracy.
4. `git push` already happened -- `git log origin/main -1` should show `2d611d6` as tip.

---

## Step-by-step log

### [2026-07-22 15:08 CDT] Starting overnight task

Scope confirmed: `001a_elastic_sphere_tissue_free_fall_test_soft` only. Will not touch
the other 9 sims or anything under `simulation/output/` besides 001a's own files.

### [2026-07-22 15:11 CDT] Step 1: commit calibration script fixes

Checked: `run_forward`'s `verbose_timing` defaulted to `True` and was never overridden
by `make_objective`/`objective`, so the `[run_forward num_frames=... setup=...s
physics=...s]` line printed on every single evaluation, not just diagnostic runs.
Fixed by:
- changing `run_forward`'s default to `verbose_timing=False`
- adding `--verbose` (store_true, default off) to argparse
- threading `verbose` through `make_objective(..., verbose=False)` -> `objective()` ->
  `run_forward(..., verbose_timing=verbose)`, and through the final full-length
  validation call in `main()`

Default output is now just the `  E=... nu=... loss=...` line per evaluation; the
timing breakdown only appears with `--verbose`.

Note: this task's Step 1 asked to commit calibrate_material.py "exactly as it stands"
with a specific message, but a nearly-identical commit (`dee3f4e`) already existed
locally from earlier in the session (made before this task list was given). Since
`dee3f4e` was never pushed to `origin/main` (confirmed via `git log origin/main..HEAD`
showing it), I amended it in place with the --verbose fix rather than leaving two
near-duplicate commits or stacking a fixup commit for something the task explicitly
said belongs in "this same commit". Resulting commit: `044f302`, same message as
specified plus one added bullet documenting the --verbose gating.

### [2026-07-22 15:16 CDT] Step 2: lightweight per-iteration trajectory logging

Added `summarize_trajectory(x, n_substeps, num_frames)`: for a (T, P, 3) substep-indexed
position array, samples at frame boundaries (`row = f * n_substeps`) and returns
per-frame centroid (mean position) and bounding-box extent (max-min), i.e.
`2 * num_frames * 3` floats -- not the full particle tensor. Sanity-checked standalone
against a synthetic array (CPU-only, no GPU needed) before wiring it in: correct shapes,
correct frame-boundary sampling offset.

Threaded through:
- `make_objective(..., n_substeps, log, trajectory_summaries, verbose=False)` -- new
  `n_substeps` and `trajectory_summaries` (mutable list, same append-in-place pattern as
  the existing `log`) params.
- `objective()` now also appends `{iteration, E, nu, loss, centroid, extent}` to
  `trajectory_summaries` for every evaluation that didn't hit a CFL violation (no valid
  trajectory to summarize for those). `iteration` is `len(log)` at call start, a shared
  counter with `optimizer_iterations_log` so a skipped CFL-violated eval doesn't shift
  indices out of alignment between the two lists.
- `main()`: builds `trajectory_summaries = []`, passes it and `n_substeps` into
  `make_objective`, and after the search computes the ground-truth centroid/extent once
  (`summarize_trajectory(gt_x.numpy(), n_substeps, full_num_frames)`, not per-evaluation).

Output JSON gains two new top-level keys alongside the existing ones:
- `trajectory_summaries`: list of per-evaluation `{iteration, E, nu, loss, centroid,
  extent}` (all evaluations, not just best-so-far -- so the viewer can show real
  Nelder-Mead exploration including backtracking).
- `ground_truth_trajectory`: `{num_frames, centroid, extent}`, computed once from
  `GtX.pt` directly (not per-particle raw data).

`optimizer_iterations_log` is unchanged (still just `E`/`nu`/`loss`/`note`) for backward
compatibility with anything already reading `001a_calibration.json`.

### [2026-07-22 15:35 CDT] Step 3: re-run on 001a -- PASSED validation

Backed up the pre-existing known-good `001a_calibration.json` to the session scratchpad
first, in case this run needed to be discarded per the task's stop condition.

Ran the exact command specified. Completed, exit code 0. Validation:
- JSON parses cleanly via `json.load` (not just assumed).
- Recovered E=14894.86 (true 14998.84, 0.69% off), nu=0.4486 (true 0.4500, 0.30% off) --
  matches the earlier known-good run almost exactly (in fact bit-identical E/nu, which
  makes sense: GT-seeding removed the only source of randomness in the pipeline, so the
  whole optimization is now fully deterministic given fixed code+config).
- `nu` shows a real progression across the 28 evaluations (0.25 -> 0.36 -> 0.40 -> 0.45
  -> ... -> 0.4486), not pinned.
- Loss descends from ~1.8e-5 to ~8.8e-10 over the run, with realistic non-monotonic
  backtracking in between (Nelder-Mead reflect/expand/contract), not a smooth fake curve.
- New `trajectory_summaries` (28 entries) and `ground_truth_trajectory` (10 frames) keys
  present with correct shapes (10 frames x 3 floats for both centroid and extent per
  entry). Spot-checked values: the best evaluation's centroid-y curve
  (`[0.546, 0.537, 0.512, ..., 0.0421]`) nearly exactly overlaps the ground-truth
  centroid-y curve (`[..., 0.0422]`); frame-0 centroid position (~0.552, 0.546, 0.0008)
  is consistent with the sphere's configured pos_x=0.55/pos_y=0.55; extent starts at
  ~0.2 (matches 2x the configured 0.1 radius) and shrinks in y at the last frame
  (consistent with floor-contact flattening).

**Verdict: PASSED. Proceeding to Step 4 (viewer panel).**

### [2026-07-22 15:20 CDT] Step 4: viewer panel added to dataset_viewer.html

Added a new sidebar section "Inverse Problem" / "001a · Calibration" (pure insertion
between the existing 5-material list and the "About this pass" footer div -- no existing
`.sim-item` entries touched) that calls a new `selectCalibrationPanel()` function,
mirroring the existing `selectMaterial()` pattern.

**Data loading deviation from the task's assumption, flagged as instructed:** the task
said to match "however the existing viewer loads its sim data" via fetch(), but on
inspection the existing viewer has **no fetch()-based loading at all** -- `MATERIALS`/
`SHARED_SCENE` are hand-authored inline JS object literals, not loaded from an external
file. There was no existing convention to match. Used `fetch()` anyway since that's
explicitly what the task asked for for *this* panel specifically, and it's the only
sensible way to load a JSON file generated by a separate script rather than hand-copying
numbers into the HTML. Fetch kicks off immediately at script-parse time (page load),
cached in a shared `CALIB_DATA_PROMISE`, independent of which sidebar item is clicked.

**Font deviation:** the task's design-language spec said "Arial", but the existing page
never uses Arial anywhere (it loads Inter/Syne/JetBrains Mono via Google Fonts and uses
them throughout). Used 'JetBrains Mono' for the new panel's numeric/data displays instead,
matching the page's actual existing convention for that kind of content (`.meta-val`,
`.json-block`, etc. all already use JetBrains Mono) rather than introducing a font the
page doesn't otherwise use. Colors: reused the page's existing `--teal`/`--orange`/
`--purple`/`--navy` CSS variables (close to, but not pixel-identical to, the hex values
in the task spec -- e.g. existing `--teal` is `#00B4A6` vs. spec's `#00BFA5`) rather than
adding near-duplicate new variables; added the one genuinely new color (`#00E5C8` bright
cyan, used only for the loss color scale) as a literal since nothing existing matches it.

**"Default to last (best) evaluation" clarification:** the task described these as
synonymous, but in the actual logged data the literal last entry isn't always the exact
loss-minimum (e.g. this run's last entry has loss=8.79e-10 while an earlier entry has
loss=7.75e-10, slightly lower). Defaulted the slider to the true argmin(loss) entry
rather than the literal last array index, since "best" is what a viewer would actually
want to see by default; in practice this landed at evaluation #26 of 28, close to the end.

Panel contents: left canvas (log-E vs. nu scatter+path, loss-colored points, orange
true-value target marker), right canvas (ground-truth solid teal vs. selected-evaluation
dashed orange centroid-height curves), a slider (0 to N-1, default = best), and a bottom
stats strip (true/recovered E and nu with % error) plus a small log-scale loss-vs-
evaluation plot. All hand-drawn via Canvas 2D (no charting library -- matches the page's
existing dependency-free approach; no `node`/npm tooling available on this machine
anyway per CLAUDE.md).

### [2026-07-22 15:27 CDT] Step 5: validation

- `001a_calibration.json`: re-confirmed valid via `json.load` (not just assumed) --
  parses cleanly, all expected top-level keys present.
- Brace/paren/bracket/backtick balance on the full `dataset_viewer.html`: `{`/`}` 231/231,
  `(`/`)` 490/490, `[`/`]` 42/42, backticks 60 (even) -- all balanced.
- No `node`/headless-chrome/playwright/selenium available on this machine (checked).
  Firefox is installed and a display is available, but a live Firefox session was already
  running under the user's own profile (X11 DISPLAY=:1) -- did **not** touch that
  (didn't want to disrupt whatever the user has open). Used a separate temporary Firefox
  profile (`-profile <scratch-dir>/ff_profile`) with `firefox --headless --screenshot`
  instead, served over a throwaway local `python3 -m http.server` on the repo root (not
  `file://`, since `fetch()` is blocked by CORS under `file://` in Firefox -- serving
  over HTTP matches how this page is actually deployed, via GitHub Pages, anyway).
  - Screenshot of the default page load: existing "001 Elastic" material view renders
    correctly, unchanged, with the new "INVERSE PROBLEM / 001a · Calibration" sidebar
    entry present alongside the original 5 untouched.
  - Screenshot of the calibration panel (via a throwaway scratch copy of the HTML with
    the init line changed to open the calibration panel instead of the default material,
    screenshotted, then deleted -- the real committed file's init line was never touched):
    fetch() loaded the JSON correctly, both canvases rendered with real data (visible
    zigzag search path with backtracking on the left, near-perfect overlap between
    ground-truth and best-fit centroid curves on the right), slider correctly defaulted
    to evaluation #26/28 (the actual best), stats strip showed E error 0.69% / nu error
    0.30% matching Step 3's numbers exactly.
- Confirmed the diff to `dataset_viewer.html` is purely additive: `git diff --stat` shows
  384 insertions, **0 deletions** -- `selectMaterial()`, `MATERIALS`, `SHARED_SCENE`, and
  all existing CSS/HTML were not modified, only new sidebar markup + new CSS rules + new
  JS functions were added, plus new-file JSON is only ever fetched, never inlined into
  the existing data structures.
- Cleaned up: killed the throwaway HTTP server, deleted the scratch test-copy HTML file
  from the repo root (was never staged/committed).

**Verdict: PASSED. Proceeding to Step 6 (docs).**

### [2026-07-22 15:32 CDT] Step 6: docs updated

Added a new CLAUDE.md section `## simulation/inverse_problem/ — material calibration via
inverse optimization (001a only)` (inserted between "What was removed" and
`dataset_viewer.html`) covering: what the script does, the repro command, the verified-
on-001a-only numbers, and the 4 known-limitations bullets (ground-truth-seeded not
video-derived, derivative-free 2-parameter only, why `initial_simplex` must be explicit,
why `--num_frames_opt` defaults to full length) plus what the new JSON keys are. Added a
bullet to the existing `dataset_viewer.html` section documenting the new panel and its
fetch()-loading deviation. Added two testing-gotcha notes (don't reuse a live Firefox
session; fetch() needs http:// not file://) since I hit both today and future sessions
would otherwise rediscover them the same way. README.md: added an `inverse_problem/`
bullet under Project Structure and a note on the `dataset_viewer.html` bullet.

Full diff (also captured to session scratchpad as `step6_docs.diff`):

```diff
diff --git a/CLAUDE.md b/CLAUDE.md
index d28276b..d7e7203 100644
--- a/CLAUDE.md
+++ b/CLAUDE.md
@@ -156,6 +156,73 @@ The `simulation/` tree used to be a full clone of `HimangiM/UniPhy_CVPR2025` (ow
 
 If you need something from one of these (e.g. a different object geometry, a different material preset, or the NCLaw training pipeline), it's gone from this repo — ask before trying to re-vendor it wholesale; a targeted re-add of just what's needed is preferable to pulling the whole UniPhy tree back in.
 
+## `simulation/inverse_problem/` — material calibration via inverse optimization (001a only)
+
+`calibrate_material.py`: recovers a sim's elastic material parameters (Young's modulus E,
+Poisson's ratio ν) by treating the repo's own forward MPM solver
+(`data_process/mpmwrapper_perparticleparams.py`) as a black box and derivative-free-optimizing
+(scipy Nelder-Mead) candidate `(E, ν)` against the sim's recorded ground-truth trajectory
+(`GtX.pt`). No NCLaw, no learned latent, no external checkpoint. Run from `data_process/` (the
+script resolves its sibling import relative to its own file location, not cwd, so this isn't
+strictly required, but keeps `--sim_dir`/`--output_json` paths short and consistent with the
+other scripts in this doc):
+```bash
+cd simulation/data_process
+conda run -n uniphy python3 ../inverse_problem/calibrate_material.py \
+  --sim_dir ../output/001a_elastic_sphere_tissue_free_fall_test_soft \
+  --output_json ../inverse_problem/results/001a_calibration.json \
+  --max_iters 15
+```
+
+**Verified on `001a` only** (2026-07-22) — E=14895 recovered vs. true 14999 (0.69% off), ν=0.4486
+vs. true 0.4500 (0.30% off) over 28 evaluations, loss descending smoothly from ~1.8e-5 to ~8.8e-10.
+Not yet run against the other 9 sims; don't assume it generalizes without checking (different
+materials have very different constitutive models — von_mises/drucker_prager/viscous_fluid don't
+even share elastic (E, ν) as their governing parameters the way `elasticity` does, so this
+specific 2-parameter script wouldn't apply as-is).
+
+**Known limitations, by design, not bugs to fix casually:**
+- **Ground-truth-seeded, not video-derived.** `run_forward()` seeds the candidate rollout's
+  particle positions/velocities directly from the ground truth's own recorded `GtX[0]`/`GtV[0]`
+  (via `wrapper.init_particles`/`wrapper.init_velocities`, overriding `MPMWrapper.__init__`'s
+  default random SDF resample) rather than from anything derived from real video — a genuine
+  RGB-video inverse problem wouldn't have this. This was a deliberate fix (see git history / this
+  file's earlier notes) to eliminate resampling noise (`np.random.choice` in
+  `mpmwrapper_perparticleparams.py`'s `MPMWrapper.__init__` has no fixed seed), which dropped the
+  loss floor by ~13 orders of magnitude at identical parameters (~6e-3 -> ~1.5e-16) — but it means
+  this script currently answers "can the solver recover parameters given the true initial state,"
+  not "can parameters be recovered from an image/video-derived initial state," a materially easier
+  problem. Safe specifically because it seeds at t=0 (F=identity, C=zero are correct there); would
+  not be safe to seed from any other frame's recorded state.
+- **Derivative-free, 2-parameter only.** Nelder-Mead over `(log E, sigmoid-squashed ν)` — no
+  gradients, doesn't scale gracefully to more parameters (would need a different optimizer, e.g.
+  something exploiting Taichi's autodiff, for materials with more than 2 free parameters like
+  von_mises's `yield_stress`/`plastic_viscosity` or drucker_prager's `friction_alpha`).
+- **`initial_simplex` must be explicit**, not scipy's default: any `x0` coordinate that's exactly
+  `0` (the `nu_raw` parameterization starts there) gets a scipy-internal tiny 0.00025 absolute
+  perturbation instead of a 5% relative one when scipy auto-builds the initial simplex, which left
+  ν completely unexplored in earlier runs regardless of rollout length — fixed by passing an
+  explicit `initial_simplex` with a deliberate, comparable step in both dimensions
+  (`step_E=0.3, step_nu=1.0`). If you add more free parameters, re-check this — scipy's default
+  simplex is unsafe for any parameterization where an initial value can land at exactly 0.
+- **`--num_frames_opt` defaults to the sim's own full length**, not a short rollout: verified via
+  direct timing (`run_forward(..., verbose_timing=True)`, or `--verbose`) that per-call cost is
+  dominated by `ti.reset()`/`ti.init()` overhead (~10-20s), not substep count — a 4-frame vs.
+  10-frame rollout showed no consistent timing difference, so there's no real speed trade-off for
+  using the full length, and the full length is needed for `ν` to be identifiable at all (floor
+  contact, where Poisson's ratio actually affects the trajectory, doesn't happen during pure
+  free-fall). A diagnostic sweep at fixed E=15000 across `ν ∈ [0.1, 0.49]` (full-length rollout)
+  showed an 8-orders-of-magnitude loss range with a razor-sharp minimum exactly at the true
+  ν=0.45 — confirms ν actually is strongly identifiable once floor contact is in the window, this
+  isn't a permanently-flat-loss physical-identifiability problem.
+- Output JSON (`simulation/inverse_problem/results/001a_calibration.json`) includes
+  `trajectory_summaries` (per-evaluation `{iteration, E, nu, loss, centroid, extent}`, lightweight
+  per-frame summaries — not the full particle tensor — for **every** evaluation including
+  non-improving probes, so the real Nelder-Mead search path including backtracking can be
+  inspected/plotted) and `ground_truth_trajectory` (the same per-frame centroid/extent computed
+  once from `GtX.pt`, so a consumer doesn't need to load `GtX.pt` directly). `dataset_viewer.html`
+  visualizes both (see below).
+
 ## `dataset_viewer.html`
 
 Standalone, dependency-free static HTML/JS page (no build step), deployed via GitHub Pages.
@@ -173,8 +240,20 @@ geometry/gravity/grid config that's identical across every sim, so it isn't dupl
   — the id-based path convention means adding a new variant only needs one `id` field, not four.
 - File paths are relative to the HTML file's own location (repo root, since GitHub Pages serves
   from there).
-
-No JS runtime (`node` etc.) is installed on this dev machine to lint/typecheck the inline script — sanity-check edits with a brace/paren/bracket balance count (and a backtick-parity count, since the template-literal-heavy render functions are the likeliest source of a subtle break) or by opening the file in a browser (`xdg-open`/`firefox dataset_viewer.html`, X11 display already available on this machine) rather than assuming a `node --check` step is available.
+- A separate sidebar section ("Inverse Problem" / "001a · Calibration",
+  `selectCalibrationPanel()`) visualizes `simulation/inverse_problem/results/001a_calibration.json`
+  — loaded via `fetch()` at page load into a shared `CALIB_DATA_PROMISE` (this is the *only*
+  fetch()-based data loading in the file; `MATERIALS`/`SHARED_SCENE` above are hand-authored
+  inline instead, so don't assume a general fetch-from-JSON convention exists beyond this one
+  panel). Hand-drawn Canvas 2D charts (no charting library): a parameter-space search-path plot
+  (log E vs. ν, points colored by loss, connected in evaluation order so backtracking is visible,
+  true value marked), a trajectory overlay (ground-truth vs. selected-evaluation centroid height
+  vs. frame), a slider over all logged evaluations (defaults to the actual argmin(loss) entry, not
+  necessarily the literal last array index), and a result-summary strip with a log-scale
+  loss-vs-evaluation plot. This panel is additive and independent of the 5-material
+  `selectMaterial()` viewer above — don't couple the two.
+
+No JS runtime (`node` etc.) is installed on this dev machine to lint/typecheck the inline script — sanity-check edits with a brace/paren/bracket balance count (and a backtick-parity count, since the template-literal-heavy render functions are the likeliest source of a subtle break) or by opening the file in a browser (`xdg-open`/`firefox dataset_viewer.html`, X11 display already available on this machine) rather than assuming a `node --check` step is available. If a Firefox session is already running under the user's own profile, don't reuse it for headless screenshot testing (`firefox --headless --screenshot` fails against an already-running instance anyway) — use a separate `-profile <scratch-dir>` instead. `fetch()` calls are blocked by CORS under `file://` — serve over `python3 -m http.server` (matches how GitHub Pages actually serves this file) rather than opening the file directly when testing anything that fetches JSON.
 
 ## `notebooks/review_trajectories.ipynb`
 
diff --git a/README.md b/README.md
index 71aa488..2d5bfcc 100644
--- a/README.md
+++ b/README.md
@@ -25,9 +25,15 @@ per-sim table, reproduction commands, and pipeline details.
     render (`renders/sphere_drop_<id>.gif`).
   - `analyze_sim.py`, `compute_diagnostics.py`, `compute_keyframes.py` — validation/analysis
     scripts run against a sim's output directory.
+  - `inverse_problem/` — `calibrate_material.py`, a derivative-free inverse-problem script
+    that recovers a sim's material parameters (E, ν) by optimizing the forward MPM solver
+    against its own recorded ground-truth trajectory. Verified on `001a` only so far; see
+    `CLAUDE.md` for scope and known limitations.
 - `notebooks/review_trajectories.ipynb` — loads a sim's raw tensors, prints shapes/dtypes/particle
   counts, and plots a quick 3D scatter + trajectory summary, for eyeballing any sim's raw data.
-- `dataset_viewer.html` — standalone dataset browser, deployed via GitHub Pages.
+- `dataset_viewer.html` — standalone dataset browser, deployed via GitHub Pages. Includes an
+  "Inverse Problem" panel visualizing the `001a` calibration search (parameter-space path,
+  trajectory overlay, convergence).
 
 ## Research Context
```

**Verdict: docs updated, proceeding to Step 7 (commit and push).**
