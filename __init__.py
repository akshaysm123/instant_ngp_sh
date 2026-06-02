"""Standalone Instant-NGP SH field.

A self-contained extraction of the Instant-NGP neural texture used in Nexels: a
multi-resolution hash grid + small MLP (via tiny-cuda-nn) that maps a 3D world
position to spherical-harmonics (SH) coefficients encoding a view-dependent color.

Typical use:

    from instant_ngp_sh import InstantNGPSHField, FieldConfig, sh_to_rgb

    field = InstantNGPSHField(aabb=[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                              config=FieldConfig(sh_degree=3))
    density, sh = field(positions)        # positions: [N, 3] world coords
    rgb = sh_to_rgb(sh, view_dirs)        # [N, 3] in [0, 1]

The SH coefficients are the model's primary output; the density head is only used by
the included volumetric trainer (``train.py``) and can be ignored or disabled
(``FieldConfig(predict_density=False)``) when you supply your own geometry.
"""

from .model import FieldConfig, InstantNGPSHField, trunc_exp
from .sh import (
    RGB2SH,
    SH2RGB,
    eval_sh,
    num_sh_coeffs,
    rgb_to_sh,
    sh_to_rgb,
)
from .rendering import (
    ray_aabb_intersect,
    render_image,
    sample_along_rays,
    volume_render_rays,
)
from .colmap import ColmapDataset, get_rays_pinhole

__all__ = [
    "InstantNGPSHField",
    "FieldConfig",
    "trunc_exp",
    "eval_sh",
    "sh_to_rgb",
    "rgb_to_sh",
    "SH2RGB",
    "RGB2SH",
    "num_sh_coeffs",
    "volume_render_rays",
    "render_image",
    "sample_along_rays",
    "ray_aabb_intersect",
    "ColmapDataset",
    "get_rays_pinhole",
]
