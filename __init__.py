"""Standalone Instant-NGP appearance field.

A self-contained, canonical Instant-NGP color field (via tiny-cuda-nn): a multi-resolution
hash grid encodes the 3D world position for a small MLP, the view direction is encoded with
spherical harmonics (SH), and a small color MLP maps the geometric feature + SH-encoded
direction to an RGB color.

It is trained as a **texture atlas for a converged Gaussian-splatting model**: per-view
depth ranges exported from the splatting give surface segments, points along them are
back-projected to world space, and the field is supervised directly with the ground-truth
pixel color (see :class:`DepthRangeDataset` and ``train.py``).

Typical use:

    from instant_ngp_sh import InstantNGPField, FieldConfig

    field = InstantNGPField(aabb=[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                            config=FieldConfig(sh_degree=4))
    rgb = field(positions, directions)   # both [N, 3] world coords / view dirs
"""

from .model import FieldConfig, InstantNGPField
from .colmap import get_rays_pinhole
from .depth_dataset import DepthRangeDataset

__all__ = [
    "InstantNGPField",
    "FieldConfig",
    "DepthRangeDataset",
    "get_rays_pinhole",
]
