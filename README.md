# Instant-NGP Appearance Field (standalone)

A self-contained, **canonical Instant-NGP color field**, implemented with NVIDIA's
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

Following the original Instant-NGP design, the **3D world position** is encoded with a
multi-resolution hash grid and fed to a small MLP (producing a geometric feature vector),
while the **view direction** is encoded with spherical harmonics (SH); the geometric
feature and the SH-encoded direction are concatenated and fed to a small *color* MLP that
outputs **a single RGB color**.

```
position (x,y,z) ─► hash grid ─► MLP ─► feature ┐
                                                ├─► color MLP ─► RGB
view dir (x,y,z) ─► SH encoding ─────────────────┘
```

The field is trained as a **texture atlas for a converged 2D Gaussian-splatting model**:
the splatting (trained on depth) exports a per-pixel depth segment where splats exist, we
back-project jittered points along those segments to world space, read the ground-truth
pixel color, and supervise `field(point, dir) -> rgb` **directly** — no volume rendering
and no density.

## What's in the box

| File | Purpose |
|------|---------|
| `model.py` | `InstantNGPField` — hash-grid + MLP and SH-direction + color MLP. Output is RGB. |
| `depth_dataset.py` | `DepthRangeDataset` — COLMAP poses + per-view splat depth ranges, sampled as `(world point, view dir, color)` triples. |
| `colmap.py` | COLMAP model readers (cameras/poses/intrinsics) + ray generation, used by `DepthRangeDataset`. |
| `train.py` | training entry point (direct point-color supervision). |
| `render.py` | render/evaluate a trained checkpoint (surface-point queries). |

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
rgb = field(positions, view_dirs)             # rgb: [N, 3] in [0, 1]
```

**The model output is RGB directly** (`rgb`, shape `[..., 3]`, sigmoid-activated). The
view direction is normalized internally and SH-encoded (`config.sh_degree`, tcnn
convention: `degree**2` features) before being fed to the color MLP, as in the original
Instant-NGP. Geometry is supplied externally (by the splatting), so the field has no
density head.

## Training (splat depth ranges)

You need two things in the **same COLMAP world frame**:

1. A COLMAP scene (`images/` + `sparse/0/{cameras,images}.bin`).
2. Per-view **depth ranges** exported from a converged 2D Gaussian-splatting model
   (`render_depth_ranges.py`; see `notes/data.md`), laid out as:

```
<model>/depth_ranges/ours_<iter>/
├── train/  DSC0001.npz  DSC0002.npz  ...   # (H, W) depth_t95, depth_t05, depth_median, alpha
└── test/   ...
```

```bash
python -m instant_ngp_sh.train \
    --data /path/to/garden \
    --depth_dir /path/to/output/garden/depth_ranges/ours_30000 \
    --out runs/garden --images_dir images_4 --split train --sh_degree 4
```

How it works:
- For each valid pixel the front/back depths `[depth_t95, depth_t05]` define a segment
  along the camera ray. Each iteration draws a jittered depth in the segment, back-projects
  it to a **world point**, and supervises the field with the ground-truth pixel color and
  the ray direction as the view direction.
- The train/test split is whichever depth-range subfolder you pass via `--split` (the
  splat export already decided it); image names are matched to COLMAP by file stem.
- Pixels with low opacity or no threshold crossing (`alpha ≤ --alpha_min`, or
  `depth_t95 == 0`) are masked out. `depth_t05 == 0` falls back to the median depth.
- The **AABB is the box containing the splat surface** (back-projected segment endpoints,
  percentile-trimmed), so the hash grid allocates resolution where the field is queried.
- `--width_tau τ` (optional) down-weights wide segments via `exp(-width/τ)` to suppress
  depth-discontinuity / transparency pixels, using only the two stored depths.

Other useful flags: `--log2_hashmap_size`, `--n_levels`,
`--base_resolution`/`--max_resolution` (hash-grid), `--batch` (points per iteration),
`--white_bg` (background fill in eval images). Periodic evaluation images and a `field.pt`
checkpoint are written to `--out`.

## Rendering / evaluation

```bash
python -m instant_ngp_sh.render --data /path/to/garden \
    --depth_dir /path/to/output/garden/depth_ranges/ours_30000 \
    --ckpt runs/garden/field.pt --split test --images_dir images_4
```

Each view is rendered by querying the field at its per-pixel **surface points** (median
depth back-projected to world space), compositing onto the background where there is no
surface. Reports per-view and mean PSNR (over surface pixels) and saves PNGs.

## Notes

- This module uses **tcnn's native `HashGrid` and `SphericalHarmonics` encodings** plus
  two fully-fused MLPs, so there is no custom CUDA extension to build — only `tcnn`. The
  hash-grid hyper-parameters (16 levels × 2 features, `2^21` table, resolutions 16→1024),
  the base MLP (64-wide, 1 hidden layer → 15-D feature), the SH direction encoding
  (degree 4 → 16 features) and the color MLP (64-wide, 2 hidden layers, ReLU, sigmoid
  output) follow the standard Instant-NGP / nerfstudio defaults.
- tcnn's `HashGrid` expects inputs in `[0, 1]³`, so positions are normalized with the
  AABB (`normalize_positions`). In the depth-range pipeline the AABB is the box containing
  the splat surface; override via the `aabb` argument for your own scenes.
