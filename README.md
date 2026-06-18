# Instant-NGP Radiance Field (standalone)

A self-contained, **canonical Instant-NGP NeRF**, implemented with NVIDIA's
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

Following the original Instant-NGP design, the **3D world position** is encoded with a
multi-resolution hash grid and fed to a small *density* MLP (producing a density plus a
geometric feature vector), while the **view direction** is encoded with spherical
harmonics (SH); the geometric feature and the SH-encoded direction are concatenated and
fed to a small *color* MLP that outputs **a single RGB color**. There is **no splatting /
surfel code** here — just the neural field, a NeRF-style volume renderer to train it, and
a loader for the **COLMAP / MipNeRF360** dataset format (`images/` + `sparse/0/*.bin`).

```
position (x,y,z) ─► hash grid ─► density MLP ─► (density, feature) ┐
                                                                   ├─► color MLP ─► RGB
view dir (x,y,z) ─► SH encoding ───────────────────────────────────┘
```

## What's in the box

| File | Purpose |
|------|---------|
| `model.py` | `InstantNGPField` — hash-grid + density MLP and SH-direction + color MLP. Output is RGB (+ a density head). |
| `sh.py` | `eval_sh`, `sh_to_rgb`, `SH2RGB`/`RGB2SH` — SH ↔ RGB helpers (not used by the field; kept for convenience). |
| `rendering.py` | `volume_render_rays` / `render_image` — minimal NeRF volume renderer with ray–AABB sampling. |
| `colmap.py` | `ColmapDataset` — reads a COLMAP model (`cameras/images/points3D.bin`, MipNeRF360 / 3DGS layout) + ray generation. |
| `train.py` | training entry point. |
| `render.py` | render/evaluate a trained checkpoint. |

To reuse the field in your own project, just copy this folder and
`from instant_ngp_sh import InstantNGPField`.

## Install

```bash
conda env create -f environment.yml
conda activate instant_ngp_sh

# tiny-cuda-nn must be built against your torch/CUDA (needs a CUDA GPU), so install it
# after activating the environment:
pip install --no-build-isolation \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## The model API

```python
import torch
from instant_ngp_sh import InstantNGPField, FieldConfig

# AABB used to normalize world positions into the unit cube.
field = InstantNGPField(
    aabb=[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
    config=FieldConfig(sh_degree=4),          # SH degree of the direction encoding
).cuda()

positions = torch.rand(4096, 3, device="cuda") * 3 - 1.5   # [N, 3] world coords
view_dirs = torch.randn(4096, 3, device="cuda")            # [N, 3] view directions
density, rgb = field(positions, view_dirs)    # density: [N,1], rgb: [N,3] in [0,1]
```

**The model output is RGB directly** (`rgb`, shape `[..., 3]`, sigmoid-activated). The
view direction is normalized internally and SH-encoded (`config.sh_degree`, tcnn
convention: `degree**2` features) before being fed to the color MLP, as in the original
Instant-NGP.

### About the density head

To train this field from posed images *alone* (a COLMAP dataset, no externally provided
geometry), volume rendering is required, which needs a density. The field therefore
includes a small density head, used only by the volume renderer.

If you already have geometry (e.g. surfels / Gaussians) you can ignore `density`, or
disable the head entirely — the field then returns RGB only:

```python
field = InstantNGPField(aabb=..., config=FieldConfig(predict_density=False))
rgb = field(positions, view_dirs)            # returns RGB only
```

## Training (COLMAP / MipNeRF360 format)

The expected layout is the COLMAP / 3DGS layout (OpenCV camera convention, per-image
intrinsics):

```
scene/
  images/                 # or images_2, images_4, images_8 (pass via --images_dir)
  sparse/0/
    cameras.bin  images.bin  points3D.bin   # .txt variants also accepted
```

```bash
python -m instant_ngp_sh.train --data /path/to/garden --out runs/garden \
    --images_dir images_4 --downscale 1 --sh_degree 4
```

Notes:
- The field trains in the **COLMAP world frame**, so it lines up with a splatting model
  trained on the *same* COLMAP (see `notes/implementation_notes.md` §9).
- By default all images are used for training. Pass `--holdout 8` to hold out every 8th
  image as the `test` split (MipNeRF360 / 3DGS convention); the rest are `train`.
- The scene **AABB is derived from the sparse point cloud**, and the renderer uses per-ray
  AABB intersection to place samples. `--near`/`--far` are auto-estimated if omitted.
- There is **no scene contraction**: the central reconstructed region trains well, but far
  background (sky, distant geometry outside the box) is only approximate.
- Lens distortion is ignored (rays assume a pinhole model). For distorted COLMAP models,
  undistort first (e.g. the 3DGS/COLMAP `image_undistorter`).

Other useful flags: `--log2_hashmap_size`, `--n_levels`,
`--base_resolution`/`--max_resolution` (hash-grid), `--white_bg` (empty space defaults to
black). Periodic evaluation images and a `field.pt` checkpoint are written to `--out`.

## Rendering / evaluation

```bash
python -m instant_ngp_sh.render --data /path/to/garden --ckpt runs/garden/field.pt --split test
```

This renders every view in the split, reports per-view and mean PSNR, and saves PNGs.

## Notes

- This module uses **tcnn's native `HashGrid` and `SphericalHarmonics` encodings** plus
  two fully-fused MLPs, so there is no custom CUDA extension to build — only `tcnn`. The
  hash-grid hyper-parameters (16 levels × 2 features, `2^21` table, resolutions 16→1024),
  the density MLP (64-wide, 1 hidden layer → density + 15-D feature), the SH direction
  encoding (degree 4 → 16 features) and the color MLP (64-wide, 2 hidden layers, ReLU,
  sigmoid output) follow the standard Instant-NGP / nerfstudio defaults.
- tcnn's `HashGrid` expects inputs in `[0, 1]³`, so positions are normalized with the
  AABB (`normalize_positions`). The AABB is a robust box around the COLMAP sparse point
  cloud. Override via the `aabb` argument for your own scenes.
- The included volume renderer is deliberately simple (plain stratified sampling, no
  occupancy grid). It intersects each ray with the scene AABB to bound sampling, but it
  does **not** implement scene contraction; for speed or properly unbounded backgrounds,
  plug the field into a more advanced sampler.
