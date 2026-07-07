# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A dataset-generation pipeline for surgical tissue material characterization from RGB video, modeled after the MASIV dataset. Summer research project; related work: DiSECt, GIC, PAC-NeRF, MASIV, UniPhy.

Most `src/` subdirectories (`src/simulate/`, `src/dataset/`, `src/cameras/`) and `data/simulations/`, `data/videos/`, `data/gifs/` are still placeholders (`.gitkeep` only) — the earlier example simulation that lived under `data/simulations/000_elastic_sphere_free_fall/` was deleted as superseded (see "Canonical simulation" below). Don't assume shared library code exists just because a directory is present — check whether it's actually populated first.

The repo's real, working code now lives under `simulation/` — a UniPhy_CVPR2025-derived pipeline (Taichi MLS-MPM) that used to be an untracked nested git clone (its own `simulation/.git/`) and has since been pruned down to only what's needed to reproduce the one canonical simulation, and folded into this repo's own git history as regular tracked files.

## Canonical simulation: `001_elastic_sphere_tissue_free_fall_test`

This is the one dataset entry the repo currently ships, and the only one `dataset_viewer.html` displays. Its full output lives at `simulation/output/001_elastic_sphere_tissue_free_fall_test/`:

- `config.yaml` — fully-resolved Hydra config (sphere geometry + elastic/tissue material + sim/train params) as saved by the run.
- `GtX.pt`, `GtV.pt` (49MB each, tracked in git) — particle position/velocity per physics substep.
- `GtC.pt`, `GtF.pt`, `GtFtmp.pt`, `GtStress.pt` (145MB each) — **gitignored** (see `simulation/.gitignore`), each over GitHub's 100MB per-file limit for regular blobs. They exist on local disk but are not committed. Regenerate them with the exact Hydra command documented in `simulation/README.md`'s "Note on excluded large files" section (same command reproduces the full run, including the smaller tracked tensors).
- `sphere_keyframes.npy` — 40-sample keyframe summary (center/extent per sample) produced by `simulation/compute_keyframes.py` from `GtX.pt`, consumed by the Blender animation script.

**Reproduction entry point:** `simulation/data_process/dataset_generate.py`, a Hydra script (`config_path='configs'`, i.e. `simulation/data_process/configs/`). It imports `mpmwrapper_perparticleparams.py` → `mpm_simulator_perparticleparams.py` (the actual Taichi MPM kernels: P2G → grid_op → G2P, corotated elasticity, sparse pointer grid) and `sdf_functions.py` → the vendored `sdf/` package (geometry SDF sampling). None of this imports `nclaw` — that was a separate, now-deleted training/latent-inference codebase (see "What was removed" below).

`simulation/data_process/configs/` has been pruned to **sphere/elastic only** — other object types (torus, box, pawn, etc.) and other sphere materials (sand, plasticine, newtonian, non_newtonian, nonhomogeneous) were deleted since only sphere+elastic is ever exercised for this simulation. The top-level `default.yaml`'s Hydra defaults list and `sphere/default.yaml`'s material default were updated accordingly (now default to `sphere` / `elastic` instead of the deleted `torus` / `sand`) so the config tree has no dangling references.

**Important dt/frame gotcha:** `GtX`/`GtF` etc. store one row per **physics substep** (`dt = 1/inv_dt = 1/4800s`), not one row per macro output frame (`frame_dt = 1/inv_frame_dt = 1/24s`, only used to derive `n_substeps = round(frame_dt/dt) = 200` per frame in `mpm_simulator_perparticleparams.py`). Any validation/analysis script building a time axis from these tensors must use the substep `dt`, not `frame_dt` — using the wrong one inflates elapsed-time-derived metrics (e.g. contact time, free-fall deviation) by ~200x. Also note frame index 0 of `GtF` is never written by `dataset_generate.py`'s save loop (`get_F(i+1, ...)` starts at index 1) — it's leftover zero-initialized memory, not real physics, and must be excluded from any max/statistics computed over `GtF`.

**Running the pipeline requires a GPU + Taichi + PyTorch+CUDA + open3d.** On this dev machine, the `uniphy` conda env already has matching deps (`torch==1.13.1+cu117`, taichi, etc. — see `simulation/environment.yml`); there is no repo-level virtualenv/lockfile, so use that conda env (`conda run -n uniphy python ...`) rather than assuming system Python has these packages.

## Blender rendering pipeline

`simulation/blender/001_elastic_sphere_tissue.blend` + `simulation/blender/scripts/01_animate_sphere.py` through `09_render_settings.py` (run in that numeric order inside Blender's Scripting tab) turn `sphere_keyframes.npy` into a photorealistic render: tissue material shader, room geometry, surface materials, three-point lighting, camera, then Cycles+OptiX render settings. See `simulation/blender/scripts/README.md` for the full per-script breakdown and the `ffmpeg` commands used to turn the rendered PNG sequence into `simulation/renders/sphere_drop.gif` (the file actually shown in `dataset_viewer.html`; the old `shape_diagnostics.png`-style PNG frame sequence itself isn't committed). The old standalone `simulation/blender/tissue_material_shader.py` was replaced by `02_tissue_material.py` in this scripts/ folder — don't recreate it.

`simulation/renders/shape_diagnostics_001.png` (bounding-box extents + deformation magnitude over time) is generated from `GtX.pt`/`GtF.pt` directly via a one-off matplotlib script, not part of the Blender pipeline or `dataset_generate.py`'s own output.

## What was removed (don't try to resurrect without asking)

The `simulation/` tree used to be a full clone of `HimangiM/UniPhy_CVPR2025` (own nested `.git`, ~1.9GB). The following were deleted as unused for reproducing the canonical simulation, confirmed via import-tracing before removal:
- `simulation/nclaw/` — separate NCLaw-based training/latent-inference codebase (`train_latent_space.py`, `gradio_demo/infer_material_latent.py` consume `GtX.pt`/`GtF.pt` as training data, but nothing in `dataset_generate.py`'s import chain touches `nclaw`).
- `simulation/gradio_demo/`, `simulation/experiments/` — unused demo app / experiment scripts + their own duplicate config trees.
- `simulation/configs/` (top-level, distinct from `simulation/data_process/configs/`) — an unrelated duplicate config tree, not read by `dataset_generate.py`.
- `simulation/third_party/warp/` — vendored copy of the `warp` package; redundant since nothing in the current pipeline needs Warp (that was the *previous*, now also-deleted, `data/simulations/000_.../simulate.py` custom Warp reimplementation — a completely different codebase from this Taichi-based `data_process` pipeline, don't conflate the two if you see references to "Warp" in old commit messages).
- `simulation/assets/`, `simulation/data_process/{elastic,newtonian,non_newtonian,plasticine,sand}.py`, `dataset_generate_all_materials.py`, `train_latent_space.py`, `setup.py` — UniPhy's own demo assets / unused per-material scripts / unused orchestrators.
- `simulation/.git/` — the nested repo itself; `simulation/` is now tracked as regular files in this repo's own `.git`.
- `data/simulations/000_elastic_sphere_free_fall/` — the old example (custom Warp reimplementation, jelly preset, different physics/material params) superseded by `001_elastic_sphere_tissue_free_fall_test`.

If you need something from one of these (e.g. a different object geometry, a different material preset, or the NCLaw training pipeline), it's gone from this repo — ask before trying to re-vendor it wholesale; a targeted re-add of just what's needed is preferable to pulling the whole UniPhy tree back in.

## `configs/metadata_template.json` — still not implemented

Reference schema for a Genesis-pipeline-style metadata file (camera arrays, MPM bounds, per-object material params). This describes a *different*, not-yet-implemented simulator convention than the current Taichi/`data_process`-based pipeline in `simulation/` — treat it as the target schema for an eventual Genesis-based pipeline, not as documentation of current output.

## `dataset_viewer.html`

Standalone, dependency-free static HTML/JS page (no build step), deployed via GitHub Pages. Simulation metadata is **hardcoded inline** in a `SIMULATIONS` JS object in the `<script>` block (not fetched at runtime) — currently one entry, `001_elastic_sphere_tissue_free_fall_test`. Adding/changing a simulation means manually editing that object plus these facts about the render template:
- The render panel uses an `<img>` tag (not `<video>`) pointing at `sim.video`, since the current render asset is an animated GIF (`simulation/renders/sphere_drop.gif`), not an mp4. Captions come from `sim.video_caption` / `sim.diagnostics_caption` fields on the object (not hardcoded strings in the template).
- The framework meta-card is labeled "Engine Version" (was "Warp Version") since this sim's engine is Taichi, not Warp; it still reads from the `sim.framework.warp_version` field name (field wasn't renamed, only the visible label).
- File paths in `SIMULATIONS` (`video`, `diagnostics_image`) are relative to the HTML file's own location (repo root, since GitHub Pages serves from there), e.g. `simulation/renders/sphere_drop.gif`.
- There's a separate, unrelated sidebar "coming soon" placeholder entry `001_elastic_sphere_poke` (different ID, not yet built) — don't confuse it with the real `001_elastic_sphere_tissue_free_fall_test` entry.

No JS runtime (`node` etc.) is installed on this dev machine to lint/typecheck the inline script — sanity-check edits with a brace/paren/bracket balance count or by opening the file in a browser (`xdg-open dataset_viewer.html`, X11 display already available on this machine) rather than assuming a `node --check` step is available.

## Working with large files in git

- No Git LFS is installed or configured in this repo (`git lfs` isn't even present as a command on this machine) — everything is committed as regular git blobs. `.gitattributes` at repo root exists but is empty; earlier LFS tracking for `*.mp4` was intentionally dropped (see commit "Store simulation video as a regular git blob") because GitHub Pages cannot serve LFS content, and the dataset viewer needs raw bytes. Do not reintroduce LFS for anything that needs to be viewable from the published `dataset_viewer.html`.
- GitHub rejects any single regular blob over 100MB. Before adding new large simulation outputs, check file sizes and gitignore anything over that limit (with a comment pointing to the regen command), following the pattern already used for `GtC.pt`/`GtF.pt`/`GtFtmp.pt`/`GtStress.pt` in `simulation/.gitignore`.
- `simulation/.gitignore` patterns are resolved relative to *its own* directory (`simulation/`), not the repo root — e.g. the entry for the large tensors is written as `output/001_.../GtC.pt`, not `simulation/output/001_.../GtC.pt`. Double-check with `git check-ignore -v <path>` after editing rather than assuming a pattern works.

## No build/test tooling yet

There is no package manifest, linter, or test suite at the repo root. `simulation/environment.yml` is a conda spec for the `data_process` pipeline specifically (Taichi + PyTorch+CUDA 11.7 + open3d + hydra-core, etc.) — the `uniphy` conda env on this dev machine already satisfies it. When verifying changes to `dataset_generate.py` or the MPM simulator, run it (via the `uniphy` env) and check the printed diagnostics plus derived validation metrics (free-fall deviation vs. analytical trajectory, floor non-penetration, deformation stretch from `GtF.pt` singular values — mind the dt/frame-0 gotchas above) rather than relying on an automated test.
