# Proxy network experiment

Separate, throwaway side-experiment (Prof. Boominathan's suggestion) testing whether
a per-trajectory descriptor plus the initial state carries enough information for a
*direct* one-shot map to the final state, skipping the timestep-by-timestep MPM
rollout entirely. This is a cheap sanity check, not a replacement for the real
pipeline, and is not wired into anything else in this repo.

## Run it

```bash
bash experiments/proxy_net/run_experiment.sh
```

Wipes and regenerates `experiments/proxy_net/output/` (`comparison.png`,
`summary.json`), and prints a summary table to stdout. Requires the `uniphy` conda
env (GPU + Taichi + PyTorch), same as the rest of `simulation/`.

## Important caveats (read before trusting any number this prints)

**There is no learned latent `z` / `g_phi` / `f_theta` in this repo.** That
architecture belongs to `simulation/nclaw/`, which was deleted in an earlier
session — confirmed via repo-wide search (no matches for `g_phi`, `f_theta`,
`latent`, `z_dim` in any code path) and confirmed no pretrained weights are
vendored here or downloaded anywhere else on this machine (no `.pth`/`.ckpt`
files, no `ckpt/` folder). The current pipeline
(`simulation/data_process/mpm_simulator_perparticleparams.py`) is a pure forward
MPM solver driven by explicit scalar material parameters (`mu`, `lam`,
`yield_stress`, `plastic_viscosity`, `friction_alpha`), not a learned latent.

So this experiment substitutes those 5 explicit parameters, read straight from each
sim's `config.yaml`, as a **stand-in conditioning vector** — not a learned latent.
A low proxy error below shows only that a network can map
`(initial state, material params) -> final state` for these two specific
trajectories. It does **not** show that a learned `z` would carry equivalent
information, and does not demonstrate generalization to an unseen material.

**N=2.** The proxy MLP is trained *and* evaluated on the same 2 trajectories —
there's no meaningful train/test split with 2 points. Low reconstruction error
mostly shows the network can memorize a discriminating-key mapping
(descriptor A → output A, descriptor B → output B), not that it generalizes.

**"Rollout baseline" is the real forward solver, not a trained network.** Since
there's no separately-trained inference path to call, the baseline re-runs the
actual existing solver (`mpm_simulator_perparticleparams.py` /
`mpmwrapper_perparticleparams.py`, imported completely unmodified) from each sim's
real recorded starting state (`GtX[0]`/`GtV[0]`, not a fresh random SDF sample),
for exactly 1999 substeps — matching how index 1999 (the last row the original save
loop ever wrote; indices 2000/2001 are unwritten zero buffer, per this repo's known
trim convention) was originally produced. Because this is a deterministic re-run of
the same physics from the same inputs, its error against `GtX[1999]` should be ~0 by
construction — that's a correctness check on the harness, not a finding. The
meaningful rollout-vs-proxy comparison here is **wall-clock time**, not accuracy.

## What's actually being compared

| | Rollout baseline | Proxy MLP |
|---|---|---|
| Input | initial state, stepped through 1999 physics substeps | initial state + 5-dim material descriptor, one forward pass |
| What it reuses | the real MPM solver, unmodified | nothing from the sim pipeline except the data |
| Trained? | no (deterministic physics) | yes, on N=2 (memorization) |

## Trajectories used

- `simulation/output/001a_elastic_sphere_tissue_free_fall_test_soft` (elastic)
- `simulation/output/002a_plasticine_sphere_free_fall_test_runny` (plasticine)

The old undifferentiated `001`/`002` sims no longer exist on disk (superseded by
paired extreme-parameter variants in an earlier session). These "a" variants were
picked arbitrarily; edit the `SIMS` dict in `proxy_net.py` to use `001b`/`002b` or
any other pair instead.

## Files

- `proxy_net.py` — the whole experiment (rollout baseline, proxy MLP, plot, report).
- `run_experiment.sh` — single entry point, cleans `output/` first.
- `output/comparison.png` — predicted-vs-ground-truth scatter (x vs. y position),
  one panel per material, three overlaid point clouds each (ground truth, rollout,
  proxy).
- `output/summary.json` — the same numbers printed to stdout, machine-readable.

Does not modify, import from as a dependency-of, or get imported by anything in
`simulation/` or the main dataset pipeline — it only reads already-generated sim
outputs and imports the solver classes for the rollout baseline.
