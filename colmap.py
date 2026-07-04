"""COLMAP reconstruction reader utilities (MipNeRF360 / 3DGS style).

Expected layout::

    scene/
    ├── images/                 # (or images_2, images_4, ... ; or pass images_dir=...)
    │   ├── 000.jpg
    │   └── ...
    └── sparse/0/
        ├── cameras.bin         # (or cameras.txt)
        └── images.bin          # (or images.txt)

This reads the COLMAP reconstruction directly, so the field lives in the **same world
coordinate frame** COLMAP defines — which is exactly the frame a Gaussian-splatting model
trained on the *same* COLMAP uses. That is what makes the "color my splatting with this
field" handoff work: a world point from the splatting can be fed straight into the field.

These helpers (camera/pose readers, intrinsics, ray generation) are consumed by
:mod:`instant_ngp_sh.depth_dataset`.

Conventions
-----------
COLMAP stores a world-to-camera transform ``(R, t)`` per image with the OpenCV camera
convention (``+x`` right, ``+y`` down, ``+z`` forward). We convert to a camera-to-world
matrix ``c2w`` (``R^T``, camera center ``-R^T t``) and generate rays in that convention.
"""

from __future__ import annotations

import os
import struct
from typing import Dict, Tuple

import numpy as np
import torch

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


def _read_model(sparse_dir: str) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """Read the COLMAP cameras and images (binary preferred, text fallback)."""
    if os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    else:
        cameras = read_cameras_text(os.path.join(sparse_dir, "cameras.txt"))
        images = read_images_text(os.path.join(sparse_dir, "images.txt"))
    return cameras, images


def get_rays_pinhole(H, W, fx, fy, cx, cy, c2w, opengl: bool = False):
    """Full-image rays for a pinhole camera.

    Args:
        H, W: image size.
        fx, fy, cx, cy: intrinsics in pixels.
        c2w: ``[4,4]`` (or ``[3,4]``) camera-to-world matrix.
        opengl: if True use the OpenGL convention (+y up, -z forward); otherwise OpenCV
            (+y down, +z forward, the COLMAP convention).

    Returns:
        ``rays_o, rays_d`` each ``[H, W, 3]`` in world space. ``rays_d`` has camera-space
        ``z = 1`` (OpenCV), so a point at view-space depth ``t`` is ``rays_o + t * rays_d``.
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
