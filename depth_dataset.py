"""Dataset for *direct point-color supervision* of the Instant-NGP field.

This dataset supplies ``(world point, view direction, color)`` triples sampled directly
along the surface segments produced by a converged 2D Gaussian-splatting model. The field
is then fit as a plain regression ``field(point, dir) -> rgb`` (no compositing).

Inputs
------
* A COLMAP reconstruction (``sparse/`` + ``images/``) defining camera poses, intrinsics
  and the world frame (read with the helpers in :mod:`colmap`).
* Per-view **depth-range** NPZs exported from the splatting (see ``notes/data.md``): one
  ``<image_stem>.npz`` per view under ``<depth_dir>/<split>/``, each holding ``(H, W)``
  float32 arrays ``depth_t95`` (front of the visible stack), ``depth_t05`` (back),
  ``depth_median`` (surface) and ``alpha`` (accumulated opacity). Depths are camera /
  view-space ``z`` (positive in front of the camera). ``0`` means "threshold not reached"
  and is masked out.

Sampling
--------
For a valid pixel, the front/back depths ``[depth_t95, depth_t05]`` define a segment
along the camera ray. A point at view-space depth ``t`` is back-projected to world space
as ``p = o + t * d`` where ``d`` is the (z-normalized) camera ray in world space and ``o``
the camera center — exactly the parametrization the renderer used, so ``t`` is the
view-space depth directly. Each training sample draws a jittered ``t`` in the segment and
takes the color from the ground-truth pixel; the view direction is the ray direction.

The segment width ``depth_t05 - depth_t95`` is returned as a per-sample confidence signal
(narrow = clean opaque surface, wide = depth discontinuity / transparency); the trainer
can down-weight wide segments without storing any per-primitive data.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .colmap import (
    _find_sparse_dir,
    _intrinsics_from_camera,
    _load_image,
    _read_model,
    _resolve,
    _warn_if_distorted,
    get_rays_pinhole,
    qvec2rotmat,
)

_DEPTH_KEYS = ("depth_t95", "depth_t05", "depth_median", "alpha")


def _release_transient_memory() -> None:
    """Return freed heap back to the OS after the load loop.

    Loading hundreds of views allocates and frees many transient buffers (npz
    decompression, dtype casts, image resizes). glibc keeps that freed memory in its
    per-thread arenas, inflating RSS by a couple of GB above the stored tensors. On a
    memory-capped host (e.g. WSL) that leftover, plus the CUDA/tcnn context allocated
    right after, can trip the OOM killer. ``malloc_trim`` hands the arenas back.
    """
    import gc

    gc.collect()
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass  # non-glibc platforms: harmless no-op


def _list_depth_npz(split_dir: str) -> dict:
    """Map ``image_stem -> npz path`` for every ``*.npz`` directly under ``split_dir``."""
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Depth-range directory not found: {split_dir}")
    out = {}
    for fn in sorted(os.listdir(split_dir)):
        if fn.endswith(".npz"):
            out[os.path.splitext(fn)[0]] = os.path.join(split_dir, fn)
    if not out:
        raise FileNotFoundError(f"No .npz depth-range files under {split_dir}")
    return out


class DepthRangeDataset:
    """Posed images + per-view depth ranges, sampled as ``(point, dir, color)`` triples.

    Use :meth:`load` to build it, :meth:`sample_points` for training batches, and
    :meth:`surface_points_for_image` for evaluation/rendering.
    """

    def __init__(
        self,
        *,
        images: torch.Tensor,    # [N, H, W, 3] uint8 in [0, 255] (memory-compact)
        c2w: torch.Tensor,       # [N, 4, 4]
        fx: torch.Tensor,        # [N]
        fy: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        near_map: torch.Tensor,  # [N, H, W] float16 front depth (depth_t95)
        far_map: torch.Tensor,   # [N, H, W] float16 back depth  (depth_t05, w/ fallbacks)
        median_map: torch.Tensor,  # [N, H, W] float16 surface depth (depth_median)
        valid: torch.Tensor,     # [N, H, W] bool
        H: int,
        W: int,
        image_names: List[str],
        bg_color: Optional[torch.Tensor],
    ):
        self.images = images
        self.c2w = c2w
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.H, self.W = int(H), int(W)
        self.image_names = image_names
        self.bg_color = bg_color

        self.near_map = near_map
        self.far_map = far_map
        self.median_map = median_map
        self.valid = valid

        # Flattened views for O(1) gathering by a single linear pixel index.
        self._hw = self.H * self.W
        self._near_flat = near_map.reshape(-1)
        self._far_flat = far_map.reshape(-1)
        self._images_flat = images.reshape(-1, 3)
        self._valid_flat = valid.reshape(-1)

        # We sample valid pixels by *rejection* against this mask rather than storing an
        # explicit list of valid linear indices. That list would be int32 over every
        # valid pixel -- >1 GB for a full high-res scene (e.g. ~400M valid pixels) -- on
        # top of the stored maps, and building it spikes memory further. Counting per
        # image (a sum over one small H*W slice each) avoids a multi-GB int upcast of the
        # whole mask.
        n_images = valid.shape[0]
        total_valid = sum(
            int(self._valid_flat[i * self._hw:(i + 1) * self._hw].sum().item())
            for i in range(n_images)
        )
        if total_valid == 0:
            raise RuntimeError("No valid pixels found across the depth-range maps.")
        self.num_valid = total_valid
        self.valid_frac = total_valid / float(n_images * self._hw)

    # -- construction ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        root: str,
        depth_dir: str,
        split: str = "train",
        images_dir: str = "images",
        alpha_min: float = 0.5,
        white_background: bool = False,
        max_images: Optional[int] = None,
        holdout_every: int = 0,
        holdout_role: str = "all",
    ) -> "DepthRangeDataset":
        """Load a COLMAP scene together with its exported depth ranges.

        Args:
            root: COLMAP scene dir (contains ``images/`` and ``sparse/``). May live in a
                completely different folder than ``depth_dir``.
            depth_dir: depth-range export dir (``.../depth_ranges/ours_<iter>``) that
                contains ``train/`` and ``test/`` subfolders, or a split folder directly.
            split: ``"train"`` or ``"test"`` (selects the matching subfolder).
            images_dir: ground-truth image subfolder (e.g. ``images_4``).
            alpha_min: minimum accumulated opacity for a pixel to count as a real surface.
            white_background: background color used by the evaluator (default black).
            max_images: optionally cap the number of views.
            holdout_every: if ``> 0``, treat every ``holdout_every``-th view (by sorted
                image name, i.e. indices ``0, N, 2N, ...``) as a held-out *test* view.
                ``0`` (default) keeps every view together.
            holdout_role: which side of the holdout split to keep. ``"train"`` keeps the
                views that are *not* held out, ``"test"`` keeps only the held-out views,
                and ``"all"`` (default) keeps everything regardless of ``holdout_every``.
        """
        split_dir = os.path.join(depth_dir, split)
        if not os.path.isdir(split_dir):
            split_dir = depth_dir  # allow pointing straight at a split folder
        npz_by_stem = _list_depth_npz(split_dir)

        sparse_dir = _find_sparse_dir(root)
        cameras, images_meta = _read_model(sparse_dir)
        meta_by_stem = {
            os.path.splitext(m["name"])[0]: m for m in images_meta.values()
        }

        stems = [s for s in sorted(npz_by_stem) if s in meta_by_stem]
        missing = [s for s in sorted(npz_by_stem) if s not in meta_by_stem]
        if missing:
            import warnings
            warnings.warn(
                f"{len(missing)} depth-range file(s) have no matching COLMAP image and "
                f"are skipped (first few: {missing[:3]}).",
                stacklevel=2,
            )
        if not stems:
            raise RuntimeError("No depth-range files matched a COLMAP image by name.")

        # Optional held-out test split: pick every N-th view (by sorted name).
        if holdout_role not in ("all", "train", "test"):
            raise ValueError(
                f"holdout_role must be 'all', 'train' or 'test'; got {holdout_role!r}."
            )
        if holdout_every and holdout_every > 0 and holdout_role != "all":
            held = set(stems[::holdout_every])
            if holdout_role == "test":
                stems = [s for s in stems if s in held]
            else:  # "train"
                stems = [s for s in stems if s not in held]
            if not stems:
                raise RuntimeError(
                    f"No views left after applying holdout_every={holdout_every} with "
                    f"holdout_role={holdout_role!r}."
                )
        if max_images is not None:
            stems = stems[:max_images]

        img_root = os.path.join(root, images_dir)
        N = len(stems)
        # Preallocated, memory-compact stores (filled per view). Storing images as uint8
        # and depths as float16 keeps a full scene (hundreds of high-res views) within a
        # few GB of RAM instead of ~10 GB; a plain list + torch.stack would additionally
        # double the peak during construction and OOM on modest machines.
        images = torch.empty(0)      # allocated once H, W are known (first view)
        near_map = far_map = median_map = valid = torch.empty(0)
        c2ws: List[np.ndarray] = []
        fxs, fys, cxs, cys = [], [], [], []
        names: List[str] = []
        target_hw: Optional[Tuple[int, int]] = None

        for i, stem in enumerate(stems):
            meta = meta_by_stem[stem]
            cam = cameras[meta["camera_id"]]
            _warn_if_distorted(cam)
            fx, fy, cx, cy = _intrinsics_from_camera(cam)

            with np.load(npz_by_stem[stem]) as npz:
                missing_keys = [k for k in _DEPTH_KEYS if k not in npz]
                if missing_keys:
                    raise KeyError(
                        f"{npz_by_stem[stem]} is missing keys {missing_keys}; "
                        f"expected {list(_DEPTH_KEYS)}."
                    )
                d_t95 = torch.from_numpy(npz["depth_t95"].astype(np.float32, copy=False))
                d_t05 = torch.from_numpy(npz["depth_t05"].astype(np.float32, copy=False))
                d_med = torch.from_numpy(npz["depth_median"].astype(np.float32, copy=False))
                alpha = torch.from_numpy(npz["alpha"].astype(np.float32, copy=False))

            H, W = d_t95.shape
            if target_hw is None:
                target_hw = (H, W)
                images = torch.empty(N, H, W, 3, dtype=torch.uint8)
                near_map = torch.empty(N, H, W, dtype=torch.float16)
                far_map = torch.empty(N, H, W, dtype=torch.float16)
                median_map = torch.empty(N, H, W, dtype=torch.float16)
                valid = torch.empty(N, H, W, dtype=torch.bool)
            elif (H, W) != target_hw:
                raise ValueError(
                    "All depth-range maps must share a resolution for batching; got "
                    f"{(H, W)} vs {target_hw} for '{stem}'."
                )

            # Ground-truth image, resized to the depth-map resolution if needed.
            rgb = torch.from_numpy(_load_image(_resolve(img_root, meta["name"])))
            h0, w0 = rgb.shape[0], rgb.shape[1]
            if (h0, w0) != (H, W):
                rgb = F.interpolate(
                    rgb.permute(2, 0, 1)[None], size=(H, W), mode="area"
                )[0].permute(1, 2, 0).contiguous()

            # Intrinsics are defined at the COLMAP camera resolution; scale to (H, W).
            sx = W / cam["width"]
            sy = H / cam["height"]

            R = qvec2rotmat(meta["qvec"])
            t = meta["tvec"]
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t

            # Build per-pixel near/far/validity.
            near = d_t95
            valid_px = (alpha > alpha_min) & (near > 0)
            # depth_t05 is often unset (0). Fall back to the median surface depth, then
            # to a degenerate (point) segment at the front depth.
            far = torch.where(d_t05 > near, d_t05, torch.where(d_med > near, d_med, near))
            median = torch.where(d_med > 0, d_med, near)

            images[i] = (rgb.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
            near_map[i] = near.to(torch.float16)
            far_map[i] = far.to(torch.float16)
            median_map[i] = median.to(torch.float16)
            valid[i] = valid_px
            c2ws.append(c2w)
            fxs.append(fx * sx); fys.append(fy * sy)
            cxs.append(cx * sx); cys.append(cy * sy)
            names.append(stem)
            # Free this view's transient buffers now so the allocator can reuse the space
            # next iteration instead of growing RSS by a couple of GB over the full scene.
            del d_t95, d_t05, d_med, alpha, rgb, near, far, median, valid_px

        H, W = target_hw
        dataset = cls(
            images=images,
            c2w=torch.from_numpy(np.stack(c2ws, 0)).float(),
            fx=torch.tensor(fxs, dtype=torch.float32),
            fy=torch.tensor(fys, dtype=torch.float32),
            cx=torch.tensor(cxs, dtype=torch.float32),
            cy=torch.tensor(cys, dtype=torch.float32),
            near_map=near_map,
            far_map=far_map,
            median_map=median_map,
            valid=valid,
            H=int(H), W=int(W),
            image_names=names,
            bg_color=(torch.ones(3) if white_background else torch.zeros(3)),
        )
        _release_transient_memory()
        return dataset

    # -- device management ----------------------------------------------------

    def to(self, device, *, images_on_device: bool = False) -> "DepthRangeDataset":
        """Move the pose/intrinsics tensors to ``device``.

        The large per-pixel maps (images, depths, validity index) stay on CPU by default
        and are gathered per batch; set ``images_on_device=True`` to keep them on GPU.
        """
        self.c2w = self.c2w.to(device)
        self.fx = self.fx.to(device); self.fy = self.fy.to(device)
        self.cx = self.cx.to(device); self.cy = self.cy.to(device)
        if self.bg_color is not None:
            self.bg_color = self.bg_color.to(device)
        if images_on_device:
            self.images = self.images.to(device)
            self._images_flat = self.images.reshape(-1, 3)
            self._near_flat = self._near_flat.to(device)
            self._far_flat = self._far_flat.to(device)
            self.valid = self.valid.to(device)
            self._valid_flat = self.valid.reshape(-1)
        return self

    @property
    def device(self) -> torch.device:
        return self.c2w.device

    def num_images(self) -> int:
        return self.images.shape[0]

    # -- internal geometry ----------------------------------------------------

    def _rays_from_pixels(self, img_idx, x, y):
        """World-space ``(origin, direction)`` for integer pixel coords (z-normalized).

        ``direction`` has camera-space ``z = 1``, so a point at view-space depth ``t`` is
        ``origin + t * direction``.
        """
        dirs_cam = torch.stack(
            [
                (x.float() + 0.5 - self.cx[img_idx]) / self.fx[img_idx],
                (y.float() + 0.5 - self.cy[img_idx]) / self.fy[img_idx],
                torch.ones_like(x, dtype=torch.float32),
            ],
            dim=-1,
        )  # OpenCV: +z forward, +y down
        R = self.c2w[img_idx, :3, :3]
        rays_d = torch.einsum("bij,bj->bi", R, dirs_cam)
        rays_o = self.c2w[img_idx, :3, 3]
        return rays_o, rays_d

    # -- training -------------------------------------------------------------

    def _sample_valid_linear(self, n: int) -> torch.Tensor:
        """Draw ``n`` random *valid* linear pixel indices by rejection.

        Rejection against the stored validity mask avoids materializing an explicit
        index over all valid pixels (which can exceed 1 GB for a full high-res scene).
        We oversample by roughly the inverse valid fraction so a single pass usually
        suffices; a bounded loop tops up any shortfall for very sparse scenes. Returned
        indices live on ``self._valid_flat``'s device (CPU by default).
        """
        vflat = self._valid_flat
        total = vflat.numel()
        frac = max(self.valid_frac, 1e-4)
        parts = []
        got = 0
        for _ in range(1000):
            if got >= n:
                break
            m = int((n - got) / frac * 1.2) + 32
            cand = torch.randint(0, total, (m,), device=vflat.device)
            keep = cand[vflat[cand]]
            if keep.numel():
                parts.append(keep)
                got += keep.numel()
        sel = parts[0] if len(parts) == 1 else torch.cat(parts)
        return sel[:n].long()

    def sample_points(self, batch_size: int):
        """Sample a batch of supervised points.

        Returns ``(points, dirs, colors, widths)`` each on :attr:`device`:

        * ``points``  ``[B, 3]`` world-space sample positions (jittered along the segment),
        * ``dirs``    ``[B, 3]`` view directions (camera -> point, unnormalized),
        * ``colors``  ``[B, 3]`` ground-truth pixel colors in ``[0, 1]``,
        * ``widths``  ``[B]``   segment width ``depth_t05 - depth_t95`` (confidence).
        """
        device = self.device
        sel = self._sample_valid_linear(batch_size)

        near = self._near_flat[sel].to(device=device, dtype=torch.float32)
        far = self._far_flat[sel].to(device=device, dtype=torch.float32)
        colors = self._images_flat[sel].to(device=device, dtype=torch.float32) / 255.0

        sel_dev = sel.to(device)
        img_idx = sel_dev // self._hw
        rem = sel_dev - img_idx * self._hw
        y = rem // self.W
        x = rem - y * self.W

        u = torch.rand(batch_size, device=device)
        t = near + (far - near) * u  # jittered depth in the segment

        rays_o, rays_d = self._rays_from_pixels(img_idx, x, y)
        points = rays_o + t[:, None] * rays_d
        widths = far - near
        return points, rays_d, colors, widths

    # -- evaluation -----------------------------------------------------------

    def surface_points_for_image(self, idx: int):
        """Per-pixel surface query for view ``idx``.

        Returns ``(points, dirs, valid)`` with shapes ``[H, W, 3]``, ``[H, W, 3]``,
        ``[H, W]`` (bool). ``points`` are the median-depth surface positions in world
        space; query the field at ``points[valid]`` with ``dirs[valid]`` to get colors.
        """
        device = self.device
        rays_o, rays_d = get_rays_pinhole(
            self.H, self.W,
            float(self.fx[idx]), float(self.fy[idx]),
            float(self.cx[idx]), float(self.cy[idx]),
            self.c2w[idx], opengl=False,
        )  # [H, W, 3]
        z = self.median_map[idx].to(device=device, dtype=torch.float32)  # [H,W] depth
        points = rays_o + z[..., None] * rays_d
        valid = self.valid[idx].to(device)
        return points, rays_d, valid

    def image(self, idx: int) -> torch.Tensor:
        """Ground-truth image ``[H, W, 3]`` float32 in ``[0, 1]`` for view ``idx``."""
        return self.images[idx].to(dtype=torch.float32) / 255.0

    # -- bounding box ---------------------------------------------------------

    def compute_aabb(self, padding: float = 0.05, max_points: int = 2_000_000) -> torch.Tensor:
        """AABB that contains the splat surface, from back-projected segment endpoints.

        Back-projects a random subset of valid pixels at both the front (``depth_t95``)
        and back (``depth_t05``) depths, then takes a percentile box (to reject floaters)
        padded by ``padding`` of the extent. Returns ``[6]`` ``(xmin..zmax)`` on CPU.
        """
        device = self.device
        k = min(self.num_valid, max_points)
        # Sample valid pixels with replacement (duplicates are harmless for a bounding
        # box) by rejection, avoiding any full index over all valid pixels.
        sel = self._sample_valid_linear(k)

        near = self._near_flat[sel].to(device=device, dtype=torch.float32)
        far = self._far_flat[sel].to(device=device, dtype=torch.float32)
        sel_dev = sel.to(device)
        img_idx = sel_dev // self._hw
        rem = sel_dev - img_idx * self._hw
        y = rem // self.W
        x = rem - y * self.W

        rays_o, rays_d = self._rays_from_pixels(img_idx, x, y)
        pts = torch.cat(
            [rays_o + near[:, None] * rays_d, rays_o + far[:, None] * rays_d], dim=0
        )

        lo = torch.quantile(pts, 0.001, dim=0)
        hi = torch.quantile(pts, 0.999, dim=0)
        extent = (hi - lo).clamp(min=1e-6)
        lo = lo - padding * extent
        hi = hi + padding * extent
        return torch.cat([lo, hi]).to(torch.float32).cpu()
