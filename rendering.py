"""Minimal NeRF-style volume renderer for an :class:`InstantNGPSHField`.

This is what lets you train the position -> SH field from posed images alone: rays
are marched through the volume, the field is queried for density + SH coefficients at
each sample, the SH coefficients are converted to RGB using the ray direction, and the
samples are alpha-composited. No external geometry is required.

The renderer is intentionally small and dependency-free (plain stratified sampling, no
occupancy grid). Pass the scene ``aabb`` (derived from the COLMAP sparse point cloud): the
sampler then uses per-ray box intersection to place samples inside the reconstructed
region. This is *not* full scene contraction, so far background is only approximate, but
the central, reconstructed content trains well. (Omitting ``aabb`` falls back to sampling
between fixed ``near``/``far``.)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import torch

from .sh import sh_to_rgb

# A field maps positions [N, 3] -> (density [N, 1], sh_coeffs [N, K, 3]).
FieldFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

# near/far may be a scalar (shared by all rays) or a per-ray [N] tensor.
Bound = Union[float, torch.Tensor]


def ray_aabb_intersect(
    rays_o: torch.Tensor, rays_d: torch.Tensor, aabb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect rays with an axis-aligned box (slab method).

    Args:
        rays_o, rays_d: ``[N, 3]`` origins / directions.
        aabb: ``[6]`` box ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

    Returns:
        ``t_near`` ``[N]``, ``t_far`` ``[N]``, ``hit`` ``[N]`` (bool). For rays that miss
        the box, ``t_near``/``t_far`` are unspecified and ``hit`` is False.
    """
    aabb = aabb.to(rays_o)
    lo, hi = aabb[:3], aabb[3:]
    # Avoid division by exactly zero for axis-parallel rays.
    d = torch.where(rays_d.abs() < 1e-8, torch.full_like(rays_d, 1e-8), rays_d)
    t0 = (lo - rays_o) / d
    t1 = (hi - rays_o) / d
    t_min = torch.minimum(t0, t1).amax(dim=-1)
    t_max = torch.maximum(t0, t1).amin(dim=-1)
    hit = t_max > torch.clamp(t_min, min=0.0)
    return t_min, t_max, hit


def _as_bound(value: Bound, n_rays: int, device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device).expand(n_rays).contiguous()
    return torch.full((n_rays,), float(value), device=device)


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: Bound,
    far: Bound,
    n_samples: int,
    perturb: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stratified sampling of points along a batch of rays.

    Args:
        rays_o, rays_d: ``[N, 3]`` ray origins and (unnormalized) directions.
        near, far: scene bounds along the ray; each may be a scalar or a ``[N]`` tensor
            (per-ray bounds, e.g. from :func:`ray_aabb_intersect`).
        n_samples: samples per ray.
        perturb: jitter sample positions within each stratum (use during training).

    Returns:
        ``pts`` ``[N, S, 3]`` sample positions and ``t_vals`` ``[N, S]`` depths.
    """
    device = rays_o.device
    n_rays = rays_o.shape[0]
    near = _as_bound(near, n_rays, device)[:, None]  # [N, 1]
    far = _as_bound(far, n_rays, device)[:, None]
    t = torch.linspace(0.0, 1.0, n_samples, device=device)[None, :]  # [1, S]
    t_vals = near * (1.0 - t) + far * t  # [N, S]

    if perturb:
        mids = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        t_vals = lower + (upper - lower) * torch.rand_like(t_vals)

    pts = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]  # [N, S, 3]
    return pts, t_vals


def volume_render_rays(
    field: FieldFn,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    *,
    near: float,
    far: float,
    n_samples: int,
    sh_degree: int,
    bg_color: Optional[torch.Tensor] = None,
    perturb: bool = False,
    aabb: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Volume-render a (small) batch of rays through ``field``.

    Args:
        field: callable ``positions[N,3] -> (density[N,1], sh_coeffs[N,K,3])``.
        rays_o, rays_d: ``[N, 3]`` ray origins / directions (directions need not be
            normalized; the SH view direction is normalized internally).
        near, far: scalar scene bounds (and the near floor / fallback for rays that miss
            ``aabb``).
        n_samples: samples per ray.
        sh_degree: SH degree to evaluate.
        bg_color: ``[3]`` background color composited behind the rays (default black).
        perturb: stratified jitter (training).
        aabb: optional ``[6]`` box. If given, samples are placed between each ray's
            entry/exit points of the box (clamped to ``near``), which concentrates
            samples on the reconstructed region for large/unbounded COLMAP scenes.

    Returns:
        dict with ``rgb`` ``[N,3]``, ``depth`` ``[N]``, ``acc`` ``[N]`` (accumulated
        opacity), and ``weights`` ``[N,S]``.
    """
    device = rays_o.device
    n_rays = rays_o.shape[0]

    if aabb is not None:
        t_min, t_max, hit = ray_aabb_intersect(rays_o, rays_d, aabb)
        ray_near = torch.where(hit, t_min.clamp(min=near), torch.full_like(t_min, near))
        ray_far = torch.where(hit, t_max, torch.full_like(t_max, far))
        ray_far = torch.maximum(ray_far, ray_near + 1e-4)
        pts, t_vals = sample_along_rays(rays_o, rays_d, ray_near, ray_far, n_samples, perturb)
    else:
        pts, t_vals = sample_along_rays(rays_o, rays_d, near, far, n_samples, perturb)

    density, sh = field(pts.reshape(-1, 3))
    density = density.reshape(n_rays, n_samples)
    sh = sh.reshape(n_rays, n_samples, -1, 3)

    # View direction is shared by all samples on a ray.
    viewdirs = rays_d[:, None, :].expand(n_rays, n_samples, 3)
    rgb = sh_to_rgb(sh, viewdirs, degree=sh_degree, clamp=True)  # [N, S, 3]

    # Distance between consecutive samples (scaled by ray length so density is in
    # world units). The final interval is set to a large value (open to infinity).
    dists = t_vals[:, 1:] - t_vals[:, :-1]
    last = torch.full_like(dists[:, :1], 1e10)
    dists = torch.cat([dists, last], dim=-1)  # [N, S]
    dists = dists * torch.norm(rays_d[:, None, :], dim=-1)

    alpha = 1.0 - torch.exp(-density * dists)  # [N, S]
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1
    )[:, :-1]
    weights = alpha * transmittance  # [N, S]

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N, 3]
    acc_map = torch.sum(weights, dim=-1)  # [N]
    depth_map = torch.sum(weights * t_vals, dim=-1)  # [N]

    if bg_color is not None:
        rgb_map = rgb_map + (1.0 - acc_map[..., None]) * bg_color.to(device)

    return {"rgb": rgb_map, "depth": depth_map, "acc": acc_map, "weights": weights}


@torch.no_grad()
def render_image(
    field: FieldFn,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    *,
    near: float,
    far: float,
    n_samples: int,
    sh_degree: int,
    bg_color: Optional[torch.Tensor] = None,
    chunk: int = 1 << 15,
    aabb: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Render a full image worth of rays in chunks (no gradients).

    ``rays_o`` / ``rays_d`` may be ``[H, W, 3]`` or ``[N, 3]``; the returned maps keep
    the leading spatial shape (``rgb`` -> ``[H, W, 3]`` or ``[N, 3]``). ``aabb`` behaves
    as in :func:`volume_render_rays`.
    """
    spatial = rays_o.shape[:-1]
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    chunks = []
    for i in range(0, rays_o.shape[0], chunk):
        out = volume_render_rays(
            field,
            rays_o[i : i + chunk],
            rays_d[i : i + chunk],
            near=near,
            far=far,
            n_samples=n_samples,
            sh_degree=sh_degree,
            bg_color=bg_color,
            perturb=False,
            aabb=aabb,
        )
        chunks.append(out)

    merged = {
        "rgb": torch.cat([c["rgb"] for c in chunks], dim=0).reshape(*spatial, 3),
        "depth": torch.cat([c["depth"] for c in chunks], dim=0).reshape(*spatial),
        "acc": torch.cat([c["acc"] for c in chunks], dim=0).reshape(*spatial),
    }
    return merged
