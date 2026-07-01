# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A dataset-generation pipeline for surgical tissue material characterization from RGB video, modeled after the MASIV dataset. Summer research project; related work: DiSECt, GIC, PAC-NeRF, MASIV, UniPhy.

The repo is early-stage: most `src/` subdirectories (`src/simulate/`, `src/dataset/`, `src/cameras/`) are placeholders (only `.gitkeep`), and simulation code currently lives directly inside each `data/simulations/<id>/` folder rather than in `src/`. Don't assume shared library code exists just because a directory is present — check whether it's actually populated first.

## Repository layout

- `data/simulations/<objectID>_<sequenceID>/` — one folder per simulation, self-contained: contains its own `simulate.py`, output `metadata.json`, rendered video, and diagnostic plots. Naming convention: `NNN_description` (e.g. `000_elastic_sphere_free_fall`).
- `data/videos/`, `data/gifs/` — intended for multi-view videos / GIFs pulled out of simulation folders (currently empty, `.gitkeep` only).
- `configs/metadata_template.json` — reference schema for a Genesis-pipeline-style metadata file (camera arrays, MPM bounds, per-object material params). Note this template describes a *different* simulator convention than the current UniPhy-based simulation (see below) — treat it as the target schema for the eventual Genesis-based pipeline, not as documentation of the present simulate.py output.
- `dataset_viewer.html` — standalone, dependency-free static HTML/JS page (no build step) for browsing simulations; deployed via GitHub Pages. Simulation metadata is **hardcoded inline** in a `SIMULATIONS` JS object in the `<script>` block (not fetched at runtime), so adding a new simulation to the viewer means manually adding an entry to that object mirroring the simulation's `metadata.json`, plus paths to its video/diagnostics image.
- `notebooks/` — placeholder for Jupyter experimentation.

## Two simulator lineages — don't conflate them

1. **Genesis-based pipeline** (original intent per README, described by `configs/metadata_template.json`): full multi-view camera rig, MPM bounds/friction params, `obj1`-style per-object material blocks. Not yet implemented in code.
2. **UniPhy-based MLS-MPM re-implementation** (current, e.g. `data/simulations/000_elastic_sphere_free_fall/simulate.py`): a from-scratch, standalone MLS-MPM solver built directly on the `warp-lang` pip package (no Genesis, no custom Warp build), closely mirroring UniPhy (CVPR 2025) / NCLaw's `nclaw/sim/mpm.py`. Runs on CUDA (`device="cuda"` is hardcoded throughout). Designed to run in Google Colab with a GPU runtime:
   ```
   pip install warp-lang -q
   python simulate.py
   ```
   Each such script is a single self-contained file: builds particle geometry, defines Warp kernels (P2G → grid_op → G2P with corotated elasticity), runs the sim loop, renders a matplotlib/ffmpeg video, computes validation diagnostics (e.g. free-fall vs. analytical trajectory, floor non-penetration), and writes `metadata.json` + `shape_diagnostics.png` alongside itself.

When adding a new simulation under this second lineage, follow the existing script's structure (config constants at top, kernel definitions, `render_video`, diagnostics plot, metadata dump) rather than inventing a new layout, and record parameter provenance in comments (which UniPhy config file / preset each constant was taken from) as the existing script does.

## Working with videos in git

- `.gitattributes` / `.gitignore` show the repo *was* using Git LFS for `*.mp4`, but the latest commit ("Store simulation video as a regular git blob") moved away from that because GitHub Pages cannot serve Git LFS content — the dataset viewer is served via GitHub Pages and needs the raw video bytes. Do not reintroduce LFS tracking for files that need to be viewable from the published `dataset_viewer.html`.
- `data/simulations/**/frames/` and `data/simulations/**/*.gif` are gitignored (per-frame PNGs and GIFs are regenerated locally, not committed); the final `.mp4` and `metadata.json` are committed.

## No build/test tooling yet

There is no package manifest, linter config, or test suite in this repo. Simulation scripts are run directly with `python <script>.py` (require a CUDA GPU + `warp-lang`, plus `numpy`, `matplotlib`, `Pillow`, and `ffmpeg` on PATH for video encoding). When verifying changes to a simulation script, run it and check the printed diagnostics (free-fall deviation, floor penetration, deformation stretch) and the generated `metadata.json`/`shape_diagnostics.png` rather than relying on an automated test.
