"""Standalone Instant-NGP radiance field.

A self-contained, canonical Instant-NGP NeRF (via tiny-cuda-nn): a multi-resolution
hash grid encodes the 3D world position for a small density MLP, the view direction is
encoded with spherical harmonics (SH), and a small color MLP maps the geometric feature
+ SH-encoded direction to an RGB color.

Typical use:

    from instant_ngp_sh import InstantNGPField, FieldConfig

    field = InstantNGPField(aabb=[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                            config=FieldConfig(sh_degree=4))
    density, rgb = field(positions, directions)   # both [N, 3] world coords / dirs

The density head is used by the included volumetric trainer (``train.py``) and can be
disabled (``FieldConfig(predict_density=False)``) to use the field purely as a
position+direction -> RGB texture when you supply your own geometry.
"""

from .model import FieldConfig, InstantNGPField, trunc_exp
from .sh import (
    RGB2SH,
    SH2RGB,
    eval_sh,
    num_sh_coeffs,
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
    "InstantNGPField",
    "FieldConfig",
    "trunc_exp",
    "eval_sh",
    "sh_to_rgb",
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
