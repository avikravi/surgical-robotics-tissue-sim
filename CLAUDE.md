# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A dataset-generation pipeline for surgical tissue material characterization from RGB video, modeled after the MASIV dataset. Summer research project; related work: DiSECt, GIC, PAC-NeRF, MASIV, UniPhy.

Most `src/` subdirectories (`src/simulate/`, `src/dataset/`, `src/cameras/`) and `data/simulations/`, `data/videos/`, `data/gifs/` are still placeholders (`.gitkeep` only) — the earlier example simulation that lived under `data/simulations/000_elastic_sphere_free_fall/` was deleted as superseded (see "Canonical simulation" below). Don't assume shared library code exists just because a directory is present — check whether it's actually populated first.

The repo's real, working code now lives under `simulation/` — a UniPhy_CVPR2025-derived pipeline (Taichi MLS-MPM) that used to be an untracked nested git clone (its own `simulation/.git/`) and has since been pruned down to only what's needed to reproduce the one canonical simulation, and folded into this repo's own git history as regular tracked files.

## Canonical simulations: `00{1,2,3,4,5}{a,b}_..._sphere_free_fall_test`

The repo ships **10 dataset entries** — the original single "middle of the road" preset per
material (`001_elastic_..._test` through `005_non_newtonian_..._test`) was superseded by **paired
extreme-parameter variants**, one `a` (one end of that material's defining parameter) and one `b`
(the opposite end), so each material's constitutive behavior is visually unmistakable rather than
"realistic but similar-looking to the other four":

| ID | Material | Model | Varied axis |
|---|---|---|---|
| `001a_..._soft` / `001b_..._stiff` | elastic | `elasticity` (Neo-Hookean) | Young's modulus E (15kPa soft/jelly vs. 3MPa rigid/stiff) |
| `002a_..._runny` / `002b_..._stiff` | plasticine | `von_mises` | yield_stress (100Pa runny putty vs. 20,000Pa stiff dough) |
| `003a_..._watery` / `003b_..._honey` | newtonian | `viscous_fluid` | shear viscosity mu (2 Pa·s watery vs. 500 Pa·s honey-like) |
| `004a_..._quicksand` / `004b_..._packed` | sand | `drucker_prager` | friction_alpha (0.02 quicksand vs. 0.6 packed pile) |
| `005a_..._thinning` / `005b_..._thickening` | non_newtonian | `von_mises` | plastic_viscosity (0.02 shear-thinning vs. 8 shear-thickening) |

`dataset_viewer.html` displays all 10, paired side-by-side per material (5 groups of 2). Each
sim's full output lives at `simulation/output/<id>/`:

- `config.yaml` — fully-resolved Hydra config (sphere geometry + material + sim/train params) as saved by the run.
- `GtX.pt`, `GtV.pt` (24-50MB depending on run, tracked in git) — particle position/velocity per physics substep.
- `GtC.pt`, `GtF.pt`, `GtFtmp.pt`, `GtStress.pt` (71-145MB depending on run) — **gitignored** (see `simulation/.gitignore`) for every sim uniformly, regardless of whether a given run happens to land under GitHub's 100MB limit, for consistency. Regenerate via the Hydra commands below (same command reproduces the full run, including the smaller tracked tensors).
- `sphere_keyframes.npy` — 40-sample keyframe summary (center/extent per sample) produced by `simulation/compute_keyframes.py <sim_dir>` from `GtX.pt`, consumed by the Blender animation script.

**Why the saved `.pt` files are much bigger than the raw data:** `dataset_generate.py` builds
`final_x = particle_x[:num_particles, :buffer_len]` — a slice + `.permute(1,0,2)` view into a much
larger preallocated buffer (`particle_x = torch.zeros([1024, buffer_len, 3])`, sized for up to 1024
particles regardless of how many a given sim actually uses). `torch.save` on a non-contiguous view
serializes the *entire underlying storage*, not just the logical slice — so a 500-particle sim still
writes a 1024-particle-sized file. This is why the very first committed `001` sim's `GtX.pt` was
already 49MB despite only ~11MB of real payload. Not fixed (matches long-standing pipeline
behavior, values load back correctly) — just don't be surprised by the file size, and don't assume
it's proportional to `num_particles`.

**Reproduction entry point:** `simulation/data_process/dataset_generate.py`, a Hydra script (`config_path='configs'`, i.e. `simulation/data_process/configs/`). It imports `mpmwrapper_perparticleparams.py` → `mpm_simulator_perparticleparams.py` (the actual Taichi MPM kernels: P2G → grid_op → G2P, per-material constitutive models, sparse pointer grid) and `sdf_functions.py` → the vendored `sdf/` package (geometry SDF sampling). None of this imports `nclaw` — that was a separate, now-deleted training/latent-inference codebase (see "What was removed" below).

Example reproduction command (elastic-soft; swap `material=`, `save_dir=`, and the
`objects.sphere.material.*` overrides per the table above — see `simulation/README.md` for the
full set of 10):
```bash
cd simulation/data_process
python dataset_generate.py objects=[sphere] objects/sphere/material=elastic \
  objects.sphere.geometry.num_particles=500 objects.sphere.material.mu=5172 \
  objects.sphere.material.lam=46552 objects.sphere.material.velocity=[0,0,0] \
  train_cfg.particles_ti_root=1024 train_cfg.cuda_chunk_size=2048 \
  visualization_cfg.num_frames=10 \
  train_cfg.save_dir=001a_elastic_sphere_tissue_free_fall_test_soft \
  train_cfg.local_dir=$(pwd)/../output
```

**Floor collider varies by material, not hardcoded to one type.** `mpmwrapper_perparticleparams.py`'s
`__init__` picks `MPMSimulator.surface_separate` (blocks penetration, but lets the material leave the
floor again — required for elastic to actually bounce, and for viscous_fluid to slide/spread instead
of freezing on contact) for `elasticity`/`viscous_fluid`, and `MPMSimulator.surface_sticky` (glues
wherever a particle first touches down) for the plastic models (`von_mises`, `drucker_prager`), since
those are supposed to keep whatever shape they yielded into. This was a real bug fix, not a stylistic
choice — the whole collider used to be hardcoded to `surface_sticky` for every material, which made a
real elastic bounce physically impossible (sticky permanently zeroes *all* velocity components, not
just the inward-normal one, the instant a particle touches y=0) and was a major reason all 5
materials used to look so similar despite very different constitutive models. Since each
`dataset_generate.py` run only ever simulates one material, this is a single scene-wide choice made
from the run's material config, not a per-particle one.

**`cuda_chunk_size` / host buffer sizing must cover every substep.** `n_substeps = round(inv_dt/24)`
per frame; the Taichi-side ring buffer (`cuda_chunk_size`) and `dataset_generate.py`'s own host-side
torch buffers (`particle_x`/`v`/`C`/`F`/`Ftmp`/`stress`) both need >= `num_frames * n_substeps + 2`
slots. The host buffers used to be hardcoded to a fixed 4096-substep size — silently corrupting GPU
memory (`CUDA_ERROR_ILLEGAL_ADDRESS`) for any material/timestep combination needing more than that
(e.g. a stiffer material forced to a smaller dt for CFL). Fixed to compute the buffer size
dynamically from `num_sim_steps * substep_gt + 2`. All 10 current sims use the standard `inv_dt=4800`
(`cuda_chunk_size=2048` is enough) — a stiffer elastic preset was tried at `inv_dt=12000`
(`n_substeps=500`) and needed `cuda_chunk_size=5120`, but ultimately wasn't used (see table above;
that stiffness level hit CFL instability instead, unrelated to the buffer-size fix).

**CFL headroom differs hugely by material** at the shared `dt=1/4800s`, `dx=0.02`: fluid/plastic
materials have 20-800x margin (wave speed dominated by a soft `kappa`/`mu`/`lam`), but pure
`elasticity` has very little (~2-10x depending on E) since its wave speed scales with
`sqrt((lam+2*mu)/rho)` directly off the (potentially very large) elastic modulus. `E=8MPa` failed
CFL for the "stiff" elastic variant — verified via bisection, and confirmed seed-sensitive right at
the threshold (particle sampling has no fixed seed, so re-running the same nominal config can
pass or fail depending on which particles get sampled) — `E=3MPa` was used instead, with a
comfortable margin at the standard timestep.

**Important dt/frame gotcha:** `GtX`/`GtF` etc. store one row per **physics substep** (`dt = 1/inv_dt`), not one row per macro output frame (`frame_dt = 1/inv_frame_dt = 1/24s`, only used to derive `n_substeps = round(frame_dt/dt)` per frame in `mpm_simulator_perparticleparams.py`). Any validation/analysis script building a time axis from these tensors must use the substep `dt`, not `frame_dt` — using the wrong one inflates elapsed-time-derived metrics (e.g. contact time, free-fall deviation) by ~200x. Also note frame index 0 of `GtF` is never written by `dataset_generate.py`'s save loop (`get_F(i+1, ...)` starts at index 1) — it's leftover zero-initialized memory, not real physics, and must be excluded from any max/statistics computed over `GtF` (`simulation/compute_diagnostics.py` and `simulation/analyze_sim.py` both handle this already — reuse them rather than re-deriving the sampling logic). Likewise, `GtX`/`GtV`'s *last two* rows are never written (the save loop's `get_x_gt`/`get_v_gt` only fill indices `0..num_sim_steps*substep_gt-1`) — trim both ends, not just GtF's start.

**Running the pipeline requires a GPU + Taichi + PyTorch+CUDA + open3d.** On this dev machine, the `uniphy` conda env already has matching deps (`torch==1.13.1+cu117`, taichi, etc. — see `simulation/environment.yml`); there is no repo-level virtualenv/lockfile, so use that conda env (`conda run -n uniphy python ...`) rather than assuming system Python has these packages.

## Blender rendering pipeline

`simulation/blender/001_elastic_sphere_tissue.blend` + `simulation/blender/scripts/01_animate_sphere.py` through `09_render_settings.py` (run in that numeric order inside Blender's Scripting tab, or headlessly — see below) turn a sim's `sphere_keyframes.npy` into a photorealistic render: tissue material shader, room geometry, surface materials, three-point lighting, camera, then Cycles+OptiX render settings. See `simulation/blender/scripts/README.md` for the full per-script breakdown and the `ffmpeg` commands used to turn the rendered PNG sequence into a `sphere_drop_<id>.gif`. The old standalone `simulation/blender/tissue_material_shader.py` was replaced by `02_tissue_material.py` in this scripts/ folder — don't recreate it.

**`01_animate_sphere.py` reads `SIM_OUTPUT_DIR` from the environment** (falling back to the 001
soft-elastic path if unset, so manual use from Blender's Scripting tab still works unchanged) —
this is what lets one `.blend` + one set of scripts render all 10 sims without editing the script
per run. A headless batch render for one sim looks like:
```bash
SIM_OUTPUT_DIR=/path/to/simulation/output/<id> \
RENDER_OUT_DIR=/path/to/simulation/renders/frame_<id> \
blender --background simulation/blender/001_elastic_sphere_tissue.blend --python render_one.py
```
where `render_one.py` just `exec()`s scripts 01,02,03,04,05,06,09 in order (matching the manual
in-Blender workflow) then calls `bpy.ops.render.render(animation=True)` with `scene.render.filepath`
set to `RENDER_OUT_DIR`.

**The black triangular wedge artifact** that used to appear where the floor met the back wall in
every render was **not** a geometry gap (verified via headless raycasting across the full frame at
several sample frames — zero misses) — it was `01_animate_sphere.py` leaving behind an unmaterialed
4x4 "Ground" placeholder plane (predating `03_room_geometry.py`'s real `RoomFloor`), exactly
coplanar with the floor at z=0. Cycles renders objects with no material as pure black, so wherever
z-fighting between the two coincident planes picked "Ground" instead of "RoomFloor" for a given
pixel, that pixel rendered black. Fixed by deleting the leftover plane creation entirely (the
cleanup loop at the top of the script still removes any pre-existing "Ground" object by name, so
older `.blend` state gets cleaned up on next run too). Don't recreate a "Ground" plane — the floor
is `RoomFloor` from `03_room_geometry.py`.

`simulation/renders/shape_diagnostics_<id>.png` (bounding-box extents + deformation magnitude over
time) is generated by `simulation/compute_diagnostics.py <sim_dir> <out_png> --title <label>
--material-kind [tensor|fluid]` — a real script now (was a one-off, uncommitted matplotlib script
for the original single 001-005). `--material-kind fluid` (only for `viscous_fluid`/newtonian) plots
`|J-1|` instead of `||F-I||`, since that material only tracks a scalar volume ratio in `F[...,0,0]`,
not a real deformation gradient. Not part of the Blender pipeline or `dataset_generate.py`'s own
output — run it separately per sim, same as `compute_keyframes.py`.

`simulation/analyze_sim.py <sim_dir> --material-kind [tensor|fluid]` computes the validation metrics
shown in `dataset_viewer.html` (free-fall deviation vs. analytical trajectory, floor contact time,
floor penetration check, final/max extents, deformation stretch) directly from `GtX.pt`/`GtF.pt`,
printed as JSON. Mind the same GtF-index-0 / GtX-last-two-rows trim as above — it's handled
internally already.

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

Standalone, dependency-free static HTML/JS page (no build step), deployed via GitHub Pages.
Restructured from a flat one-sim-at-a-time layout to a **paired-comparison layout**: the data model
is a `MATERIALS` JS object keyed by 5 material names (`elastic`, `plasticine`, `newtonian`, `sand`,
`nonNewtonian`), each with a `variants: {a, b}` pair (not a flat `SIMULATIONS` object keyed by sim
ID like the original single-sim version) plus a shared `modelNote`/`axis` description of the
constitutive model and which parameter was pushed to opposite extremes. The sidebar has one
clickable entry per *material* (5 total, not 10) — clicking renders both variants side-by-side via
`.pair-grid` (falls back to stacked on narrow viewports). `SHARED_SCENE` factors out the
geometry/gravity/grid config that's identical across every sim, so it isn't duplicated 10 times.
- Each variant card's render panel uses two `<img>` tags (GIF + diagnostics PNG side by side), paths
  built from the variant's `id` field (`simulation/renders/sphere_drop_<id>.gif` /
  `shape_diagnostics_<id>.png>`) rather than separate `video`/`diagnostics_image` fields per variant
  — the id-based path convention means adding a new variant only needs one `id` field, not four.
- File paths are relative to the HTML file's own location (repo root, since GitHub Pages serves
  from there).
- There's a separate, unrelated sidebar "coming soon" placeholder entry `001_elastic_sphere_poke`
  (different ID, not yet built) — don't confuse it with the real material entries.

No JS runtime (`node` etc.) is installed on this dev machine to lint/typecheck the inline script — sanity-check edits with a brace/paren/bracket balance count (and a backtick-parity count, since the template-literal-heavy render functions are the likeliest source of a subtle break) or by opening the file in a browser (`xdg-open`/`firefox dataset_viewer.html`, X11 display already available on this machine) rather than assuming a `node --check` step is available.

## Working with large files in git

- No Git LFS is installed or configured in this repo (`git lfs` isn't even present as a command on this machine) — everything is committed as regular git blobs. `.gitattributes` at repo root exists but is empty; earlier LFS tracking for `*.mp4` was intentionally dropped (see commit "Store simulation video as a regular git blob") because GitHub Pages cannot serve LFS content, and the dataset viewer needs raw bytes. Do not reintroduce LFS for anything that needs to be viewable from the published `dataset_viewer.html`.
- GitHub rejects any single regular blob over 100MB. Before adding new large simulation outputs, check file sizes and gitignore anything over that limit (with a comment pointing to the regen command), following the pattern already used for `GtC.pt`/`GtF.pt`/`GtFtmp.pt`/`GtStress.pt` in `simulation/.gitignore`.
- `simulation/.gitignore` patterns are resolved relative to *its own* directory (`simulation/`), not the repo root — e.g. the entry for the large tensors is written as `output/001_.../GtC.pt`, not `simulation/output/001_.../GtC.pt`. Double-check with `git check-ignore -v <path>` after editing rather than assuming a pattern works.

## No build/test tooling yet

There is no package manifest, linter, or test suite at the repo root. `simulation/environment.yml` is a conda spec for the `data_process` pipeline specifically (Taichi + PyTorch+CUDA 11.7 + open3d + hydra-core, etc.) — the `uniphy` conda env on this dev machine already satisfies it. When verifying changes to `dataset_generate.py` or the MPM simulator, run it (via the `uniphy` env) and check the printed diagnostics plus derived validation metrics (free-fall deviation vs. analytical trajectory, floor non-penetration, deformation stretch from `GtF.pt` singular values — mind the dt/frame-0 gotchas above) rather than relying on an automated test.
