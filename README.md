# Surgical Robotics Tissue Simulation

A dataset generation pipeline for surgical tissue material characterization from RGB video.
Modeled after the MASIV dataset. The working pipeline is a Taichi MLS-MPM simulator adapted from
UniPhy (CVPR 2025), living under `simulation/`.

Browse the dataset at the live [dataset viewer](https://avikravi.github.io/surgical-robotics-tissue-sim/dataset_viewer.html)
(or open `dataset_viewer.html` locally — it's a standalone, dependency-free page).

## Canonical simulations

10 dataset entries — one `a`/`b` extreme-parameter pair per material (elastic, plasticine,
newtonian, sand, non-Newtonian), each a sphere free-falling onto a floor boundary, simulated with
per-particle MPM, validated against analytical free fall and floor non-penetration, and rendered
photorealistically in Blender (tissue material, room/lighting/camera). See `CLAUDE.md` for the full
per-sim table, reproduction commands, and pipeline details.

## Project Structure

- `simulation/` — the working simulation pipeline (Taichi MLS-MPM, adapted from UniPhy_CVPR2025):
  - `data_process/` — the dataset-generation pipeline (`dataset_generate.py` and the MPM kernels
    it calls) that produces the 10 canonical sims under `output/`.
  - `output/` — each sim's raw tensors (`GtX.pt`/`GtV.pt`/etc.), config, and derived keyframes.
  - `blender/` — the Blender scene + scripts that turn a sim's keyframes into a photorealistic
    render (`renders/sphere_drop_<id>.gif`).
  - `analyze_sim.py`, `compute_diagnostics.py`, `compute_keyframes.py` — validation/analysis
    scripts run against a sim's output directory.
  - `inverse_problem/` — `calibrate_material.py`, a derivative-free inverse-problem script
    that recovers a sim's material parameters (E, ν) by optimizing the forward MPM solver
    against its own recorded ground-truth trajectory. Verified on `001a` only so far; see
    `CLAUDE.md` for scope and known limitations.
- `notebooks/review_trajectories.ipynb` — loads a sim's raw tensors, prints shapes/dtypes/particle
  counts, and plots a quick 3D scatter + trajectory summary, for eyeballing any sim's raw data.
- `dataset_viewer.html` — standalone dataset browser, deployed via GitHub Pages. Includes an
  "Inverse Problem" panel visualizing the `001a` calibration search (parameter-space path,
  trajectory overlay, convergence).

## Research Context

Summer research project on tissue material characterization from RGB video during robotic surgery.
Related work: DiSECt, GIC, PAC-NeRF, MASIV, UniPhy.
