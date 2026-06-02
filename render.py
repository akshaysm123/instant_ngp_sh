"""Render a trained Instant-NGP SH field on a COLMAP split and report PSNR.

    python -m instant_ngp_sh.render --data /path/to/garden --ckpt runs/garden/field.pt --split test

Requires a CUDA GPU and tiny-cuda-nn.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "instant_ngp_sh"

import argparse
import math
import os

import torch

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio

from .colmap import ColmapDataset
from .model import FieldConfig, InstantNGPSHField
from .rendering import render_image


def mse_to_psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained Instant-NGP SH field")
    p.add_argument("--data", required=True,
                   help="scene dir with images/ and sparse/0/*.bin")
    p.add_argument("--ckpt", required=True, help="path to field.pt checkpoint")
    p.add_argument("--out", default=None, help="output dir (default: alongside ckpt)")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--downscale", type=float, default=2.0)
    p.add_argument("--images_dir", default="images", help="image subfolder")
    p.add_argument("--holdout", type=int, default=8, help="eval holdout stride")
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; tiny-cuda-nn requires a CUDA GPU.")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = FieldConfig(**ckpt["config"])
    field = InstantNGPSHField(aabb=ckpt["aabb"], config=config).to(device)
    field.load_state_dict(ckpt["state_dict"])
    field.eval()

    train_args = ckpt.get("args", {})
    white_bg = train_args.get("white_bg", False)
    near = train_args.get("near", None)
    far = train_args.get("far", None)

    dataset = ColmapDataset.load(
        args.data,
        split=args.split,
        white_background=white_bg,
        downscale=args.downscale,
        max_images=args.max_images,
        near=near,
        far=far,
        images_dir=args.images_dir,
        holdout=args.holdout,
    ).to(device)

    # Scene-box sampling, matching training.
    render_aabb = field.aabb

    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), f"render_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    psnrs = []
    for idx in range(dataset.num_images()):
        rays_o, rays_d = dataset.rays_for_image(idx)
        out = render_image(
            field,
            rays_o.to(device),
            rays_d.to(device),
            near=dataset.near,
            far=dataset.far,
            n_samples=args.n_samples,
            sh_degree=config.sh_degree,
            bg_color=dataset.bg_color.to(device) if dataset.bg_color is not None else None,
            aabb=render_aabb,
        )
        gt = dataset.images[idx].to(device)
        mse = torch.mean((out["rgb"] - gt) ** 2).item()
        psnr = mse_to_psnr(mse)
        psnrs.append(psnr)

        rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        imageio.imwrite(os.path.join(out_dir, f"{idx:03d}.png"), rgb)
        print(f"  view {idx:03d}: PSNR={psnr:.2f}")

    mean_psnr = sum(psnrs) / len(psnrs)
    print(f"\n{args.split} mean PSNR over {len(psnrs)} views: {mean_psnr:.2f}")
    print(f"Renders written to {out_dir}")


if __name__ == "__main__":
    main()
