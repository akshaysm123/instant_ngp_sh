# Instant-NGP SH Field (standalone)

A self-contained extraction of the **Instant-NGP neural texture** used in
*Nexels: Neurally-Textured Surfels*. It is a multi-resolution hash-grid encoding
(Instant-NGP) followed by a small fully-fused MLP, implemented with NVIDIA's
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

The field maps a **3D world position → spherical-harmonics (SH) coefficients** that
encode a view-dependent color. A helper converts those coefficients into an RGB color
given a viewing direction. There is **no splatting / surfel code** here — just the
neural field, a NeRF-style volume renderer to train it, and a loader for the
**COLMAP / MipNeRF360** dataset format (`images/` + `sparse/0/*.bin`).

```
position (x,y,z) ──► hash grid ──► MLP ──► SH coefficients ──(+ view dir)──► RGB
```

## What's in the box

| File | Purpose |
|------|---------|
| `model.py` | `InstantNGPSHField` — the hash grid + MLP. Output is SH coefficients (+ an optional density head). |
| `sh.py` | `sh_to_rgb`, `eval_sh`, `rgb_to_sh`, `SH2RGB`/`RGB2SH` — SH ↔ RGB conversions. |
| `rendering.py` | `volume_render_rays` / `render_image` — minimal NeRF volume renderer with ray–AABB sampling. |
| `colmap.py` | `ColmapDataset` — reads a COLMAP model (`cameras/images/points3D.bin`, MipNeRF360 / 3DGS layout) + ray generation. |
| `train.py` | training entry point. |
| `render.py` | render/evaluate a trained checkpoint. |

To reuse the field in your own project, just copy this folder and
`from instant_ngp_sh import InstantNGPSHField, sh_to_rgb`.

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
from instant_ngp_sh import InstantNGPSHField, FieldConfig, sh_to_rgb

# AABB used to normalize world positions into the unit cube.
field = InstantNGPSHField(
    aabb=[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
    config=FieldConfig(sh_degree=3),          # 16 SH coeffs -> 48 output channels
).cuda()

positions = torch.rand(4096, 3, device="cuda") * 3 - 1.5   # [N, 3] world coords
density, sh = field(positions)                # density: [N,1], sh: [N, 16, 3]

view_dirs = torch.randn(4096, 3, device="cuda")
rgb = sh_to_rgb(sh, view_dirs, degree=3)      # [N, 3] in [0, 1]
```

**The model output is the SH color** (`sh`, shape `[..., K, 3]` with `K=(deg+1)**2`).
`sh_to_rgb` applies the Nexels/3DGS convention `rgb = clamp(0.5 + eval_sh(sh, dir), 0, 1)`.

### About the density head

To train this field from posed images *alone* (a COLMAP dataset, no externally provided
geometry), volume rendering is required, which needs a density. The field therefore
includes a small density head, used only by the volume renderer.

If you already have geometry (e.g. surfels / Gaussians) you can ignore `density` and use
only the SH output, or disable the head entirely:

```python
field = InstantNGPSHField(aabb=..., config=FieldConfig(predict_density=False))
sh = field(positions)            # returns SH coefficients only
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
    --images_dir images_4 --downscale 1 --holdout 8 --sh_degree 3
```

Notes:
- The field trains in the **COLMAP world frame**, so it lines up with a splatting model
  trained on the *same* COLMAP (see `notes/implementation_notes.md` §9).
- Every `--holdout`-th image (default 8, the MipNeRF360 convention) is held out as the
  `test` split; the rest are `train`.
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

## Notes & differences from Nexels

- Nexels uses a **custom CUDA hash-grid** kernel (with a surfel-specific antialiasing
  term) plus the tcnn MLP. This standalone module uses **tcnn's native `HashGrid`
  encoding** instead, so there is no custom CUDA extension to build — only `tcnn`. The
  hash-grid hyper-parameters (16 levels × 2 features, `2^21` table, resolutions 16→1024)
  and the MLP (64-wide, 2 hidden layers, ReLU) match the Nexels defaults.
- tcnn's `HashGrid` expects inputs in `[0, 1]³`, so positions are normalized with the
  AABB (`normalize_positions`). The AABB is a robust box around the COLMAP sparse point
  cloud. Override via the `aabb` argument for your own scenes.
- The included volume renderer is deliberately simple (plain stratified sampling, no
  occupancy grid). It intersects each ray with the scene AABB to bound sampling, but it
  does **not** implement scene contraction; for speed or properly unbounded backgrounds,
  plug the field into a more advanced sampler.
