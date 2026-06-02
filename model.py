"""Standalone Instant-NGP field that maps a 3D world position to SH coefficients.

This is the same architecture used as the *neural texture* in Nexels: a
multi-resolution hash-grid encoding (Instant-NGP) followed by a small fully-fused
MLP, implemented with NVIDIA's ``tiny-cuda-nn`` (tcnn). The MLP outputs spherical
harmonics (SH) coefficients that encode a view-dependent color; use
:func:`instant_ngp_sh.sh.sh_to_rgb` to turn them into an RGB color given a view
direction.

The field optionally also predicts a volumetric density. The density head is only
needed to *train the field directly from posed images* (see ``rendering.py`` /
``train.py``): without externally provided geometry, volume rendering is the only way
to supervise a position -> color field from a NeRF/MipNeRF-style dataset. If you
already have geometry (e.g. surfels / Gaussians), set ``predict_density=False`` (or
simply ignore the returned density) and use the SH output directly.

Positions are normalized into the unit cube using an axis-aligned bounding box
(AABB) before being fed to the hash grid, which expects inputs in ``[0, 1]^3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .sh import num_sh_coeffs


class _TruncExp(torch.autograd.Function):
    """Exponential with gradients clamped for numerical stability (Instant-NGP).

    Used as the density activation: the forward pass is a plain ``exp`` (clamped to
    avoid ``inf``), while the backward pass clamps the exponent to ``[-15, 15]`` so a
    large/empty-space density does not produce an exploding gradient.
    """

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        ctx.save_for_backward(x)
        return torch.exp(x.clamp(max=15.0))

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        return grad * torch.exp(x.clamp(-15.0, 15.0))


def trunc_exp(x: torch.Tensor) -> torch.Tensor:
    return _TruncExp.apply(x)


_DENSITY_ACTIVATIONS = {
    "trunc_exp": trunc_exp,
    "exp": torch.exp,
    "softplus": torch.nn.functional.softplus,
    "relu": torch.nn.functional.relu,
}


@dataclass
class FieldConfig:
    """Configuration for :class:`InstantNGPSHField`.

    Defaults mirror the Nexels neural-texture settings.
    """

    sh_degree: int = 3
    # Hash grid (Instant-NGP) settings.
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 21
    base_resolution: int = 16          # "minres" in Nexels
    max_resolution: int = 1024         # "maxres" in Nexels
    # MLP settings.
    mlp_hidden_dim: int = 64
    mlp_num_hidden_layers: int = 2
    # Density head (only used by the volumetric trainer).
    predict_density: bool = True
    density_activation: str = "trunc_exp"
    density_bias: float = -1.0         # added to the raw density logit before activation


class InstantNGPSHField(nn.Module):
    """Instant-NGP hash-grid + MLP field: world position -> SH coefficients.

    Args:
        aabb: ``[6]`` or ``[2, 3]`` axis-aligned bounding box
            ``(xmin, ymin, zmin, xmax, ymax, zmax)`` used to normalize positions to
            ``[0, 1]^3``. Points are clamped to this box.
        config: a :class:`FieldConfig` (or ``None`` for defaults).

    Forward:
        ``forward(positions)`` where ``positions`` is ``[..., 3]`` in world space.

        Returns ``(density, sh_coeffs)`` if ``config.predict_density`` else
        ``sh_coeffs``, where

        * ``density``   is ``[..., 1]`` (non-negative),
        * ``sh_coeffs`` is ``[..., K, 3]`` with ``K = (sh_degree + 1) ** 2``.
    """

    def __init__(self, aabb, config: Optional[FieldConfig] = None):
        super().__init__()
        self.config = config or FieldConfig()
        cfg = self.config

        aabb = torch.as_tensor(aabb, dtype=torch.float32).reshape(-1)
        if aabb.numel() != 6:
            raise ValueError("aabb must have 6 elements (xmin,ymin,zmin,xmax,ymax,zmax)")
        self.register_buffer("aabb", aabb)

        self.num_coeffs = num_sh_coeffs(cfg.sh_degree)
        self.sh_dim = 3 * self.num_coeffs
        self.density_dim = 1 if cfg.predict_density else 0
        self.n_output_dims = self.density_dim + self.sh_dim

        if cfg.density_activation not in _DENSITY_ACTIVATIONS:
            raise ValueError(
                f"density_activation must be one of {list(_DENSITY_ACTIVATIONS)}"
            )
        self._density_act = _DENSITY_ACTIVATIONS[cfg.density_activation]

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
        self.network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": cfg.mlp_hidden_dim,
            "n_hidden_layers": cfg.mlp_num_hidden_layers,
        }

        self.model = _build_tcnn_module(
            n_input_dims=3,
            n_output_dims=self.n_output_dims,
            encoding_config=self.encoding_config,
            network_config=self.network_config,
        )

    # -- public API -----------------------------------------------------------

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Map world-space ``positions`` (``[..., 3]``) into ``[0, 1]^3`` via the AABB."""
        aabb_min = self.aabb[:3]
        aabb_max = self.aabb[3:]
        normalized = (positions - aabb_min) / (aabb_max - aabb_min)
        return normalized.clamp(0.0, 1.0)

    def forward(
        self, positions: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cfg = self.config
        batch_shape = positions.shape[:-1]
        x = self.normalize_positions(positions).reshape(-1, 3)

        raw = self.model(x).float()  # tcnn returns fp16; SH math wants fp32

        if cfg.predict_density:
            density_logit = raw[..., 0:1] + cfg.density_bias
            density = self._density_act(density_logit)
            sh = raw[..., 1:]
        else:
            density = None
            sh = raw

        sh = sh.reshape(*batch_shape, self.num_coeffs, 3)
        if cfg.predict_density:
            density = density.reshape(*batch_shape, 1)
            return density, sh
        return sh

    def get_sh(self, positions: torch.Tensor) -> torch.Tensor:
        """Return only the SH coefficients ``[..., K, 3]`` (ignores density)."""
        out = self.forward(positions)
        return out[1] if self.config.predict_density else out

    def density(self, positions: torch.Tensor) -> torch.Tensor:
        """Return only the density ``[..., 1]`` (requires ``predict_density=True``)."""
        if not self.config.predict_density:
            raise RuntimeError("field was built with predict_density=False")
        return self.forward(positions)[0]


def _build_tcnn_module(n_input_dims, n_output_dims, encoding_config, network_config):
    """Construct the fused tcnn hash-grid + MLP module.

    Imported lazily so the rest of the package (SH utils, dataset, the volume
    renderer with a custom field) can be used on machines without tcnn installed.
    """
    try:
        import tinycudann as tcnn
    except ImportError as exc:  # pragma: no cover - depends on the user's environment
        raise ImportError(
            "tiny-cuda-nn (tcnn) is required for InstantNGPSHField. Install it with:\n"
            "  pip install --no-build-isolation "
            "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch\n"
            "It requires a CUDA toolkit and a CUDA-capable GPU."
        ) from exc

    return tcnn.NetworkWithInputEncoding(
        n_input_dims=n_input_dims,
        n_output_dims=n_output_dims,
        encoding_config=encoding_config,
        network_config=network_config,
    )
