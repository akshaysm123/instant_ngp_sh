"""Loader for the COLMAP dataset format (MipNeRF360 / 3DGS style).

Expected layout::

    scene/
    ├── images/                 # (or images_2, images_4, ... ; or pass images_dir=...)
    │   ├── 000.jpg
    │   └── ...
    └── sparse/0/
        ├── cameras.bin         # (or cameras.txt)
        ├── images.bin          # (or images.txt)
        └── points3D.bin        # (or points3D.txt; optional, used for the scene AABB)

This reads the COLMAP reconstruction directly, so the field trains in the **same world
coordinate frame** COLMAP defines — which is exactly the frame a 3D Gaussian Splatting /
surfel model trained on the *same* COLMAP uses. That is what makes the
"color my splatting with this field" handoff (see ``notes/implementation_notes.md`` §9)
work: a world point from the splatting can be fed straight into the field.

Conventions
-----------
COLMAP stores a world-to-camera transform ``(R, t)`` per image with the OpenCV camera
convention (``+x`` right, ``+y`` down, ``+z`` forward). We convert to a camera-to-world
matrix ``c2w`` (``R^T``, camera center ``-R^T t``) and generate rays in that convention.

Unbounded scenes
----------------
MipNeRF360 scenes are unbounded. This module does **not** implement scene contraction;
instead the renderer concentrates samples inside an axis-aligned box derived from the
sparse point cloud (see ``compute_aabb``), using per-ray box intersection for near/far.
The central, reconstructed region trains well; far background is only approximate. 
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:  # imageio v2 API (avoids deprecation warnings on newer versions)
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore


def _load_image(path: str) -> np.ndarray:
    """Load an image as a float32 RGB array in ``[0, 1]`` with shape ``[H, W, 3]``."""
    img = np.asarray(imageio.imread(path), dtype=np.float32) / 255.0
    if img.ndim == 2:  # grayscale -> RGB
        img = np.stack([img, img, img], axis=-1)
    return img[..., :3]  # drop alpha if present (COLMAP images are opaque)


# COLMAP camera model id -> (name, number of parameters).
_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
# Models whose first parameter is a single shared focal length.
_SIMPLE_FOCAL_MODELS = {
    "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "FOV",
    "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
}


def qvec2rotmat(qvec) -> np.ndarray:
    """COLMAP quaternion ``(w, x, y, z)`` -> 3x3 rotation matrix (world-to-camera R)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Binary readers                                                              #
# --------------------------------------------------------------------------- #

def _read(fid, num_bytes, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(num_bytes))


def read_cameras_binary(path: str) -> Dict[int, dict]:
    cameras = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            cam_id, model_id, width, height = _read(f, 24, "iiQQ")
            name, n_params = _CAMERA_MODELS[model_id]
            params = _read(f, 8 * n_params, "d" * n_params)
            cameras[cam_id] = dict(model=name, width=int(width), height=int(height),
                                   params=np.array(params, dtype=np.float64))
    return cameras


def read_images_binary(path: str) -> Dict[int, dict]:
    images = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            d = _read(f, 64, "idddddddi")
            image_id, qvec, tvec, cam_id = d[0], d[1:5], d[5:8], d[8]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            num_2d = _read(f, 8, "Q")[0]
            f.seek(24 * num_2d, os.SEEK_CUR)  # skip the 2D keypoints
            images[image_id] = dict(qvec=np.array(qvec), tvec=np.array(tvec),
                                    camera_id=cam_id, name=name.decode())
    return images


def read_points3D_binary(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        xyz = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            d = _read(f, 43, "QdddBBBd")
            xyz[i] = d[1:4]
            track_len = _read(f, 8, "Q")[0]
            f.seek(8 * track_len, os.SEEK_CUR)  # skip the track
    return xyz


# --------------------------------------------------------------------------- #
# Text readers (fallback)                                                     #
# --------------------------------------------------------------------------- #

def read_cameras_text(path: str) -> Dict[int, dict]:
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            e = line.split()
            cam_id = int(e[0])
            model = e[1]
            cameras[cam_id] = dict(model=model, width=int(e[2]), height=int(e[3]),
                                   params=np.array([float(v) for v in e[4:]]))
    return cameras


def read_images_text(path: str) -> Dict[int, dict]:
    images = {}
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    # Image entries occupy two lines each; the second (keypoints) is ignored.
    for i in range(0, len(lines), 2):
        e = lines[i].split()
        image_id = int(e[0])
        qvec = np.array([float(v) for v in e[1:5]])
        tvec = np.array([float(v) for v in e[5:8]])
        cam_id = int(e[8])
        name = e[9]
        images[image_id] = dict(qvec=qvec, tvec=tvec, camera_id=cam_id, name=name)
    return images


def read_points3D_text(path: str) -> np.ndarray:
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            e = line.split()
            pts.append([float(e[1]), float(e[2]), float(e[3])])
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3))


def _intrinsics_from_camera(cam: dict) -> Tuple[float, float, float, float]:
    """Extract ``(fx, fy, cx, cy)`` from a COLMAP camera dict (distortion ignored)."""
    p = cam["params"]
    if cam["model"] in _SIMPLE_FOCAL_MODELS:
        fx = fy = float(p[0])
        cx, cy = float(p[1]), float(p[2])
    else:  # PINHOLE / OPENCV / FULL_OPENCV / ...
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    return fx, fy, cx, cy


def _find_sparse_dir(root: str) -> str:
    for candidate in (os.path.join(root, "sparse", "0"), os.path.join(root, "sparse")):
        if os.path.exists(os.path.join(candidate, "cameras.bin")) or \
           os.path.exists(os.path.join(candidate, "cameras.txt")):
            return candidate
    raise FileNotFoundError(
        f"No COLMAP model found under {root} (looked for sparse/0/ and sparse/)."
    )


def _read_model(sparse_dir: str):
    if os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    else:
        cameras = read_cameras_text(os.path.join(sparse_dir, "cameras.txt"))
        images = read_images_text(os.path.join(sparse_dir, "images.txt"))
    xyz = None
    for ext, reader in ((".bin", read_points3D_binary), (".txt", read_points3D_text)):
        pp = os.path.join(sparse_dir, "points3D" + ext)
        if os.path.exists(pp):
            xyz = reader(pp)
            break
    return cameras, images, xyz


@dataclass
class ColmapDataset:
    """Posed images from a COLMAP reconstruction (OpenCV camera convention).

    Provides ``num_images``, ``sample_rays``, ``rays_for_image``, ``compute_aabb`` and
    ``to`` (consumed by ``train.py`` / ``render.py``), with **per-image** intrinsics
    ``(fx, fy, cx, cy)`` and a world frame identical to the COLMAP reconstruction.
    """

    images: torch.Tensor   # [N, H, W, 3]
    c2w: torch.Tensor      # [N, 4, 4]
    fx: torch.Tensor       # [N]
    fy: torch.Tensor       # [N]
    cx: torch.Tensor       # [N]
    cy: torch.Tensor       # [N]
    H: int
    W: int
    near: float
    far: float
    bg_color: Optional[torch.Tensor]
    point_xyz: Optional[torch.Tensor] = None  # sparse points, for the AABB

    @property
    def focal(self) -> float:  # representative value, for logging
        return float(self.fx[0])

    @classmethod
    def load(
        cls,
        root: str,
        split: str = "train",
        downscale: float = 1.0,
        images_dir: str = "images",
        holdout: int = 0,
        white_background: bool = False,
        near: Optional[float] = None,
        far: Optional[float] = None,
        max_images: Optional[int] = None,
    ) -> "ColmapDataset":
        """Load a COLMAP scene.

        Args:
            root: scene directory containing ``images/`` and ``sparse/``.
            split: ``"train"``, ``"test"`` (or ``"all"``). With ``holdout > 0``, test is
                every ``holdout``-th image (MipNeRF360 / 3DGS convention) and train is the rest.
            downscale: resize images by this factor (intrinsics are scaled to match).
            images_dir: image subfolder (e.g. ``"images_4"`` for the 4x downsampled set).
            holdout: hold out every N-th image for the test split (default 0 = no holdout).
            white_background: composite color for empty space (default black).
            near, far: ray bounds; if ``None`` they are auto-estimated from the scene.
            max_images: optionally cap the number of images.
        """
        sparse_dir = _find_sparse_dir(root)
        cameras, images_meta, xyz = _read_model(sparse_dir)

        items = sorted(images_meta.values(), key=lambda d: d["name"])
        if holdout and holdout > 0 and split in ("train", "test"):
            keep_train = lambda i: (i % holdout) != 0
            items = [it for i, it in enumerate(items)
                     if keep_train(i) == (split == "train")]
        if max_images is not None:
            items = items[:max_images]
        if len(items) == 0:
            raise RuntimeError(f"No images for split '{split}' (holdout={holdout}).")

        img_root = os.path.join(root, images_dir)
        imgs: List[torch.Tensor] = []
        c2ws: List[np.ndarray] = []
        fxs, fys, cxs, cys = [], [], [], []
        target_hw: Optional[Tuple[int, int]] = None

        for it in items:
            cam = cameras[it["camera_id"]]
            fx, fy, cx, cy = _intrinsics_from_camera(cam)
            _warn_if_distorted(cam)

            # Pose: COLMAP stores world-to-camera (R, t); we want camera-to-world.
            # x_c = R * x_w + t   -->   R^T (x_c -t) = x_w
            # c2w =     [ R^T   -t ] 
            #           [ 0     1  ] 
            R = qvec2rotmat(it["qvec"])
            t = it["tvec"]
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t

            rgb = torch.from_numpy(_load_image(_resolve(img_root, it["name"])))  # [h0,w0,3]
            h0, w0 = rgb.shape[0], rgb.shape[1]

            # Intrinsics are defined for the COLMAP camera resolution; scale to the
            # actually-loaded image size (handles images_N folders) and the downscale.
            sx = w0 / cam["width"]
            sy = h0 / cam["height"]
            new_h, new_w = int(round(h0 / downscale)), int(round(w0 / downscale))
            if (new_h, new_w) != (h0, w0):
                rgb = F.interpolate(rgb.permute(2, 0, 1)[None], size=(new_h, new_w),
                                    mode="area")[0].permute(1, 2, 0).contiguous()
            sx *= new_w / w0
            sy *= new_h / h0

            if target_hw is None:
                target_hw = (new_h, new_w)
            elif (new_h, new_w) != target_hw:
                raise ValueError(
                    "All images must share a resolution for batching; got "
                    f"{(new_h, new_w)} vs {target_hw}. Use a single images_dir."
                )

            imgs.append(rgb)
            c2ws.append(c2w)
            fxs.append(fx * sx); fys.append(fy * sy)
            cxs.append(cx * sx); cys.append(cy * sy)

        images = torch.stack(imgs, 0).float()
        c2w = torch.from_numpy(np.stack(c2ws, 0)).float()
        fx = torch.tensor(fxs, dtype=torch.float32)
        fy = torch.tensor(fys, dtype=torch.float32)
        cx = torch.tensor(cxs, dtype=torch.float32)
        cy = torch.tensor(cys, dtype=torch.float32)
        H, W = target_hw

        point_xyz = (torch.from_numpy(xyz).float()
                     if xyz is not None and len(xyz) > 0 else None)

        near_v, far_v = _auto_near_far(c2w, point_xyz, near, far)
        bg = torch.ones(3) if white_background else torch.zeros(3)

        return cls(images=images, c2w=c2w, fx=fx, fy=fy, cx=cx, cy=cy,
                   H=int(H), W=int(W), near=near_v, far=far_v, bg_color=bg,
                   point_xyz=point_xyz)

    # -- interface consumed by train.py / render.py --------------------------

    def num_images(self) -> int:
        return self.images.shape[0]

    def to(self, device) -> "ColmapDataset":
        self.images = self.images.to(device)
        self.c2w = self.c2w.to(device)
        self.fx = self.fx.to(device); self.fy = self.fy.to(device)
        self.cx = self.cx.to(device); self.cy = self.cy.to(device)
        if self.bg_color is not None:
            self.bg_color = self.bg_color.to(device)
        if self.point_xyz is not None:
            self.point_xyz = self.point_xyz.to(device)
        return self

    def sample_rays(self, batch_size, generator=None):
        device = self.images.device
        n = self.num_images()
        # sample random pixels in random images
        img_idx = torch.randint(0, n, (batch_size,), generator=generator, device=device)
        y = torch.randint(0, self.H, (batch_size,), generator=generator, device=device)
        x = torch.randint(0, self.W, (batch_size,), generator=generator, device=device)
        rgb = self.images[img_idx, y, x]

        dirs = torch.stack(
            [
                # + 0.5 for pixel center
                (x.float() + 0.5 - self.cx[img_idx]) / self.fx[img_idx],
                (y.float() + 0.5 - self.cy[img_idx]) / self.fy[img_idx],
                torch.ones(batch_size, device=device),
            ],
            dim=-1,
        )  # OpenCV: +z forward, +y down
        c2w = self.c2w[img_idx]
        rays_d = torch.einsum("bij,bj->bi", c2w[:, :3, :3], dirs)
        rays_o = c2w[:, :3, 3]
        return rays_o, rays_d, rgb

    def rays_for_image(self, idx: int):
        return get_rays_pinhole(
            self.H, self.W,
            float(self.fx[idx]), float(self.fy[idx]),
            float(self.cx[idx]), float(self.cy[idx]),
            self.c2w[idx], opengl=False,
        )

    def compute_aabb(self, padding: float = 0.1) -> torch.Tensor:
        """Robust AABB around the scene, from the sparse points if available.

        Uses the 0.5-99.5 percentile of the point cloud to ignore COLMAP outliers, then
        pads by ``padding`` of the extent. Falls back to the camera positions.
        """
        if self.point_xyz is not None and self.point_xyz.shape[0] >= 8:
            pts = self.point_xyz
            lo = torch.quantile(pts, 0.005, dim=0)
            hi = torch.quantile(pts, 0.995, dim=0)
        else:
            cams = self.c2w[:, :3, 3]
            lo = cams.min(0).values
            hi = cams.max(0).values
        extent = (hi - lo).clamp(min=1e-6)
        lo = lo - padding * extent
        hi = hi + padding * extent
        return torch.cat([lo, hi]).to(torch.float32).cpu()


def get_rays_pinhole(H, W, fx, fy, cx, cy, c2w, opengl: bool = False):
    """Full-image rays for a pinhole camera.

    Args:
        H, W: image size.
        fx, fy, cx, cy: intrinsics in pixels.
        c2w: ``[4,4]`` (or ``[3,4]``) camera-to-world matrix.
        opengl: if True use the OpenGL convention (+y up, -z forward); otherwise OpenCV
            (+y down, +z forward, the COLMAP convention).

    Returns:
        ``rays_o, rays_d`` each ``[H, W, 3]`` in world space.
    """
    device = c2w.device
    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    x = (i + 0.5 - cx) / fx
    y = (j + 0.5 - cy) / fy
    if opengl:
        dirs = torch.stack([x, -y, -torch.ones_like(x)], dim=-1)
    else:
        dirs = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    rays_d = torch.einsum("ij,hwj->hwi", c2w[:3, :3], dirs)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o, rays_d


def _auto_near_far(c2w, point_xyz, near, far) -> Tuple[float, float]:
    """Estimate sensible scalar near/far from scene scale (used as clamps/fallbacks).

    Per-ray near/far comes from AABB intersection in the renderer; these scalars only
    bound rays that miss the box and provide a near floor.
    """
    if point_xyz is not None and point_xyz.shape[0] >= 8:
        lo = torch.quantile(point_xyz, 0.005, dim=0)
        hi = torch.quantile(point_xyz, 0.995, dim=0)
        center = 0.5 * (lo + hi)
        radius = 0.5 * float(torch.linalg.norm(hi - lo))
    else:
        cams = c2w[:, :3, 3]
        center = cams.mean(0)
        radius = float(torch.linalg.norm(cams - center, dim=1).max())
    radius = max(radius, 1e-3)
    cam_dists = torch.linalg.norm(c2w[:, :3, 3] - center, dim=1)
    near_v = near if near is not None else max(1e-3, 0.02 * radius)
    far_v = far if far is not None else float(cam_dists.max()) + 2.0 * radius
    return float(near_v), float(far_v)


def _resolve(img_root: str, name: str) -> str:
    path = os.path.join(img_root, name)
    if os.path.exists(path):
        return path
    base = os.path.splitext(name)[0]
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        if os.path.exists(os.path.join(img_root, base + ext)):
            return os.path.join(img_root, base + ext)
    raise FileNotFoundError(f"Image '{name}' not found under {img_root}")


def _warn_if_distorted(cam: dict) -> None:
    """Warn once if a camera carries non-trivial lens distortion (ignored by the rays)."""
    p = cam["params"]
    n_pinhole = 4 if cam["model"] not in _SIMPLE_FOCAL_MODELS else 3
    if len(p) > n_pinhole and np.any(np.abs(p[n_pinhole:]) > 1e-8):
        import warnings
        warnings.warn(
            f"COLMAP camera model '{cam['model']}' has distortion params that are "
            f"ignored (rays assume a pinhole model). Undistort the images first "
            f"(e.g. the 3DGS/COLMAP image_undistorter) for best results.",
            stacklevel=2,
        )
