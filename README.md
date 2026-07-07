# Surgical Robotics Tissue Simulation

A dataset generation pipeline for surgical tissue material characterization from RGB video.
Modeled after the MASIV dataset. The eventual target is a Genesis-based multi-view pipeline
(see `configs/metadata_template.json` for the planned schema); the current, working pipeline
is a Taichi MLS-MPM simulator adapted from UniPhy (CVPR 2025), living under `simulation/`.

Browse the dataset at the live [dataset viewer](https://avikravi.github.io/surgical-robotics-tissue-sim/dataset_viewer.html)
(or open `dataset_viewer.html` locally — it's a standalone, dependency-free page).

## Canonical simulation

`simulation/output/001_elastic_sphere_tissue_free_fall_test/` — a soft-tissue-parameterized
elastic sphere free-falling onto a floor boundary, simulated with per-particle MPM (corotated
elasticity), validated against analytical free fall and floor non-penetration, and rendered
photorealistically in Blender (tissue material, room/lighting/camera).
See `CLAUDE.md` for the reproduction command and pipeline details.

## Project Structure

- `simulation/` — the actual working simulation pipeline (Taichi MLS-MPM, adapted from
  UniPhy_CVPR2025), the canonical `001_...` simulation output, and the Blender rendering
  scripts that turn it into `simulation/renders/sphere_drop.gif`.
- `data/simulations/`, `data/videos/`, `data/gifs/` — placeholders (`.gitkeep` only) for the
  future Genesis-based pipeline's per-simulation outputs.
- `src/simulate/`, `src/dataset/`, `src/cameras/` — placeholders for the future Genesis-based
  simulation, data-saving, and multi-view camera code.
- `configs/` — `metadata_template.json`, the target metadata schema for the Genesis pipeline.
- `notebooks/` — Jupyter notebooks for experimentation.
- `dataset_viewer.html` — standalone dataset browser, deployed via GitHub Pages.

## Research Context

Summer research project on tissue material characterization from RGB video during robotic surgery.
Related work: DiSECt, GIC, PAC-NeRF, MASIV, UniPhy.
