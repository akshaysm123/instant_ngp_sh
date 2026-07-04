"""Canonical Instant-NGP field: ``(world position, view direction) -> RGB``.

This is the standard Instant-NGP / nerfstudio appearance field:

* the 3D **position** is encoded with a multi-resolution hash grid (Instant-NGP)
  and fed to a small MLP that outputs a geometric feature vector;
* the **view direction** is encoded with spherical harmonics (SH);
* the geometric feature and the SH-encoded direction are concatenated and fed to
  a small *color* MLP that outputs a single ``RGB`` color (sigmoid-activated).

Everything is implemented with NVIDIA's ``tiny-cuda-nn`` (tcnn) fully-fused
encodings and MLPs. Here SH is used as the *input encoding of the view direction*
(as in the original Instant-NGP paper), not as the network output: the field emits
RGB directly.

There is no density head: geometry is supplied externally (e.g. by a converged
Gaussian-splatting model), so the field is used purely as a position+direction ->
RGB texture atlas and trained by direct point-color supervision (see ``train.py``).

Positions are normalized into the unit cube using an axis-aligned bounding box
(AABB) before being fed to the hash grid, which expects inputs in ``[0, 1]^3``.
Directions are expected to be unit vectors; they are mapped to ``[0, 1]^3`` before
the SH encoding, following the tcnn / nerfstudio convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FieldConfig:
    """Configuration for :class:`InstantNGPField`.

    Defaults mirror the standard Instant-NGP / nerfstudio appearance field.
    """

    # Hash grid (Instant-NGP) position encoding.
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 21
    base_resolution: int = 16          # "minres"
    max_resolution: int = 1024         # "maxres"
    # Spherical-harmonics view-direction encoding. tcnn produces ``sh_degree ** 2``
    # features (e.g. degree 4 -> 16), matching the original Instant-NGP.
    sh_degree: int = 4
    # Position MLP: hash features -> geometric feature vector.
    geo_feat_dim: int = 15
    mlp_hidden_dim: int = 64
    mlp_num_hidden_layers: int = 1
    # Color MLP: (geo feature + SH-encoded direction) -> RGB.
    color_mlp_hidden_dim: int = 64
    color_mlp_num_hidden_layers: int = 2


class InstantNGPField(nn.Module):
    """Instant-NGP appearance field: ``(position, direction) -> RGB``.

    Args:
        aabb: ``[6]`` or ``[2, 3]`` axis-aligned bounding box
            ``(xmin, ymin, zmin, xmax, ymax, zmax)`` used to normalize positions to
            ``[0, 1]^3``. Points are clamped to this box.
        config: a :class:`FieldConfig` (or ``None`` for defaults).

    Forward:
        ``forward(positions, directions)`` where both are ``[..., 3]`` (positions in
        world space, directions are view directions, normalized internally). Returns
        ``rgb`` ``[..., 3]`` in ``[0, 1]``.
    """

    def __init__(self, aabb, config: Optional[FieldConfig] = None):
        super().__init__()
        self.config = config or FieldConfig()
        cfg = self.config

        aabb = torch.as_tensor(aabb, dtype=torch.float32).reshape(-1)
        if aabb.numel() != 6:
            raise ValueError("aabb must have 6 elements (xmin,ymin,zmin,xmax,ymax,zmax)")
        self.register_buffer("aabb", aabb)

        self.geo_feat_dim = cfg.geo_feat_dim

        # Geometric growth factor between hash-grid levels (Instant-NGP).
        if cfg.n_levels > 1:
            per_level_scale = (cfg.max_resolution / cfg.base_resolution) ** (
                1.0 / (cfg.n_levels - 1)
            )
        else:
            per_level_scale = 1.0
        self.per_level_scale = per_level_scale

        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": cfg.n_levels,
            "n_features_per_level": cfg.n_features_per_level,
            "log2_hashmap_size": cfg.log2_hashmap_size,
            "base_resolution": cfg.base_resolution,
            "per_level_scale": per_level_scale,
        }
        self.direction_encoding_config = {
            "otype": "SphericalHarmonics",
            "degree": cfg.sh_degree,
        }
        self.base_network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": cfg.mlp_hidden_dim,
            "n_hidden_layers": cfg.mlp_num_hidden_layers,
        }
        self.color_network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",
            "n_neurons": cfg.color_mlp_hidden_dim,
            "n_hidden_layers": cfg.color_mlp_num_hidden_layers,
        }

        (
            self.position_encoding,
            self.direction_encoding,
            self.mlp_base,
            self.mlp_head,
        ) = _build_tcnn_modules(
            geo_feat_dim=self.geo_feat_dim,
            encoding_config=self.encoding_config,
            direction_encoding_config=self.direction_encoding_config,
            base_network_config=self.base_network_config,
            color_network_config=self.color_network_config,
        )

    # -- public API -----------------------------------------------------------

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Map world-space ``positions`` into ``[0, 1]^3`` via the AABB. Query points
        outside the AABB are clamped to the box surface.
        """
        aabb_min = self.aabb[:3]
        aabb_max = self.aabb[3:]
        normalized = (positions - aabb_min) / (aabb_max - aabb_min)
        return normalized.clamp(0.0, 1.0)

    def forward(
        self, positions: torch.Tensor, directions: torch.Tensor
    ) -> torch.Tensor:
        batch_shape = positions.shape[:-1]

        x = self.normalize_positions(positions).reshape(-1, 3)
        geo_feat = self.mlp_base(self.position_encoding(x)).float()

        dirs = torch.nn.functional.normalize(directions.reshape(-1, 3), dim=-1)
        # tcnn's SH encoding expects directions mapped from [-1, 1] to [0, 1].
        dir_feat = self.direction_encoding((dirs + 1.0) * 0.5)

        rgb = self.mlp_head(torch.cat([geo_feat, dir_feat], dim=-1)).float()
        return rgb.reshape(*batch_shape, 3)


def _build_tcnn_modules(
    geo_feat_dim,
    encoding_config,
    direction_encoding_config,
    base_network_config,
    color_network_config,
):
    """Construct the fused tcnn encodings + MLPs for the appearance field.

    Returns ``(position_encoding, direction_encoding, mlp_base, mlp_head)``.

    Imported lazily so the rest of the package (dataset utilities) can be used on
    machines without tcnn installed.
    """
    try:
        import tinycudann as tcnn
    except ImportError as exc:  # pragma: no cover - depends on the user's environment
        raise ImportError(
            "tiny-cuda-nn (tcnn) is required for InstantNGPField. Install it with:\n"
            "  pip install --no-build-isolation "
            "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch\n"
            "It requires a CUDA toolkit and a CUDA-capable GPU."
        ) from exc

    position_encoding = tcnn.Encoding(
        n_input_dims=3, encoding_config=encoding_config
    )
    direction_encoding = tcnn.Encoding(
        n_input_dims=3, encoding_config=direction_encoding_config
    )
    mlp_base = tcnn.Network(
        n_input_dims=position_encoding.n_output_dims,
        n_output_dims=geo_feat_dim,
        network_config=base_network_config,
    )
    # Color MLP input: geometric feature vector + SH-encoded view direction.
    mlp_head = tcnn.Network(
        n_input_dims=geo_feat_dim + direction_encoding.n_output_dims,
        n_output_dims=3,
        network_config=color_network_config,
    )
    return position_encoding, direction_encoding, mlp_base, mlp_head
