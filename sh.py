"""Spherical harmonics utilities.

The Instant-NGP field in this package outputs *SH coefficients* (a view-independent
representation of a view-dependent color). These helpers turn those coefficients into
an RGB color given a viewing direction, and provide the DC-only conversions between
RGB and SH.

Coefficient layout
-------------------
Coefficients are stored coeff-major with the color channel last::

    coeffs[..., k, c]   k in [0, (deg+1)**2),  c in {0,1,2} == R,G,B

which matches the layout used by the original Nexels / 3DGS texture path
(``mlp_out[..., :3*K].reshape(..., K, 3)``).

RGB convention
--------------
Following 3DGS / Nexels, the final color includes a constant ``0.5`` offset::

    rgb = 0.5 + eval_sh(deg, coeffs, dirs)

so that a freshly initialized (near-zero) field produces mid-gray rather than black.
"""

from __future__ import annotations

import torch

# Real spherical harmonics constants (same values used by 3DGS / PlenOctrees).
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]

MAX_SH_DEGREE = 3


def num_sh_coeffs(degree: int) -> int:
    """Number of SH basis functions for a given degree, ``(degree + 1) ** 2``."""
    return (degree + 1) ** 2


def eval_sh(degree: int, coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate an SH expansion at unit directions.

    Args:
        degree: SH degree in ``[0, 3]``.
        coeffs: ``[..., K, 3]`` SH coefficients, ``K >= (degree + 1) ** 2``.
        dirs:   ``[..., 3]`` viewing directions (need not be normalized, but
                should be for a meaningful result).

    Returns:
        ``[..., 3]`` evaluated value (this is *without* the ``0.5`` RGB offset;
        use :func:`sh_to_rgb` for a clamped RGB color).
    """
    if not 0 <= degree <= MAX_SH_DEGREE:
        raise ValueError(f"degree must be in [0, {MAX_SH_DEGREE}], got {degree}")
    k = num_sh_coeffs(degree)
    if coeffs.shape[-2] < k:
        raise ValueError(
            f"coeffs has {coeffs.shape[-2]} basis functions but degree {degree} "
            f"requires at least {k}"
        )

    result = C0 * coeffs[..., 0, :]
    if degree >= 1:
        x = dirs[..., 0:1]
        y = dirs[..., 1:2]
        z = dirs[..., 2:3]
        result = (
            result
            - C1 * y * coeffs[..., 1, :]
            + C1 * z * coeffs[..., 2, :]
            - C1 * x * coeffs[..., 3, :]
        )
        if degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * coeffs[..., 4, :]
                + C2[1] * yz * coeffs[..., 5, :]
                + C2[2] * (2.0 * zz - xx - yy) * coeffs[..., 6, :]
                + C2[3] * xz * coeffs[..., 7, :]
                + C2[4] * (xx - yy) * coeffs[..., 8, :]
            )
            if degree >= 3:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * coeffs[..., 9, :]
                    + C3[1] * xy * z * coeffs[..., 10, :]
                    + C3[2] * y * (4 * zz - xx - yy) * coeffs[..., 11, :]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * coeffs[..., 12, :]
                    + C3[4] * x * (4 * zz - xx - yy) * coeffs[..., 13, :]
                    + C3[5] * z * (xx - yy) * coeffs[..., 14, :]
                    + C3[6] * x * (xx - 3 * yy) * coeffs[..., 15, :]
                )
    return result


def sh_to_rgb(
    coeffs: torch.Tensor,
    dirs: torch.Tensor,
    degree: int = MAX_SH_DEGREE,
    clamp: bool = True,
) -> torch.Tensor:
    """Convert SH coefficients + viewing direction into an RGB color in ``[0, 1]``.

    ``rgb = 0.5 + eval_sh(degree, coeffs, dirs)`` (then optionally clamped to
    ``[0, 1]``), matching the Nexels / 3DGS convention.

    Args:
        coeffs: ``[..., K, 3]`` SH coefficients (the model output).
        dirs:   ``[..., 3]`` viewing directions; will be normalized internally.
        degree: SH degree to evaluate (``<=`` the degree the field was built with).
        clamp:  if True, clamp the result to ``[0, 1]``.

    Returns:
        ``[..., 3]`` RGB color.
    """
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    rgb = eval_sh(degree, coeffs, dirs) + 0.5
    if clamp:
        rgb = rgb.clamp(0.0, 1.0)
    return rgb


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Inverse of the DC (constant) SH term: ``(rgb - 0.5) / C0``.

    Useful for initializing the DC coefficient of an SH field from a base color.
    """
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    """DC-only SH coefficient -> RGB (``sh * C0 + 0.5``)."""
    return sh * C0 + 0.5


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """RGB -> DC-only SH coefficient (``(rgb - 0.5) / C0``)."""
    return (rgb - 0.5) / C0
