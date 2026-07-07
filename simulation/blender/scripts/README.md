# Blender Rendering Pipeline — 001_elastic_sphere_tissue

Run these scripts in order, inside Blender's Scripting tab, against
`001_elastic_sphere_tissue.blend`, to reproduce the photorealistic render
of the MPM elastic sphere free-fall simulation.

The underlying physics (MLS-MPM via UniPhy/Taichi) is computed entirely
upstream of Blender — see `simulation/data_process/dataset_generate.py`
and `simulation/compute_keyframes.py`. Blender performs zero physics; it
only visualizes precomputed keyframe data (`sphere_keyframes.npy`).

| Script | Purpose |
|---|---|
| `01_animate_sphere.py` | Loads keyframes, creates + animates TissueSphere and Ground |
| `02_tissue_material.py` | Subtle organic tissue shader (color/roughness variation, no bump) |
| `03_room_geometry.py` | Floor, back/left walls, table (height matched to sphere rest position); also purges any leftover surgical tool objects from older runs |
| `04_surface_materials.py` | Wood table, light blue walls, tile floor |
| `05_lighting.py` | Three-point area light rig (key/fill/rim), tuned to avoid overexposure |
| `06_camera.py` | HeroCamera, ~24%-frame-height sphere framing, tracks table center |
| `09_render_settings.py` | Cycles + OptiX, 384 samples, denoising, AgX, world ambient |

After running all scripts, render the full animation with **Ctrl+F12**
(PNG sequence to `simulation/renders/`), then convert to GIF:

```bash
cd simulation/renders/frame_
ffmpeg -framerate 25 -i %04d.png -vf "palettegen" palette.png -y
ffmpeg -framerate 25 -i %04d.png -i palette.png -lavfi "paletteuse" -loop 0 sphere_drop.gif -y
```
