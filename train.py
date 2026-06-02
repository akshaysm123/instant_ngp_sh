"""Train the Instant-NGP SH field on a COLMAP (MipNeRF360 / 3DGS) scene.

The field maps a 3D world position to SH coefficients (a view-dependent color). To
supervise it from posed images alone, we volume-render rays (using the field's density
head) and minimize the photometric error against the ground-truth pixels. The scene AABB
(from the COLMAP sparse point cloud) bounds per-ray sampling.

Run as a module (recommended)::

    python -m instant_ngp_sh.train --data /path/to/garden --out runs/garden \
        --images_dir images_4 --downscale 1

The data directory must contain ``images/`` and ``sparse/0/{cameras,images,points3D}.bin``
(use a downsampled image folder such as ``images_4`` for large scenes). Requires a CUDA
GPU and tiny-cuda-nn.
"""

from __future__ import annotations

# -- allow running both as `python -m instant_ngp_sh.train` and as a plain script ----
if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "instant_ngp_sh"

import argparse
import math
import os
import time

import torch

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio

from .colmap import ColmapDataset
from .model import FieldConfig, InstantNGPSHField
from .rendering import render_image, volume_render_rays


def mse_to_psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def parse_args():
    p = argparse.ArgumentParser(description="Train Instant-NGP SH field on a COLMAP scene")
    # Data.
    p.add_argument("--data", required=True,
                   help="scene dir with images/ and sparse/0/{cameras,images,points3D}.bin")
    p.add_argument("--out", default="runs/ngp_sh", help="output directory")
    p.add_argument("--downscale", type=float, default=2.0, help="image downscale factor")
    p.add_argument("--images_dir", default="images",
                   help="image subfolder, e.g. images_4 for the 4x downsampled set")
    p.add_argument("--holdout", type=int, default=8,
                   help="hold out every N-th image for eval (MipNeRF360 convention)")
    p.add_argument("--white_bg", action="store_true", default=False,
                   help="composite empty space onto white (default black)")
    p.add_argument("--near", type=float, default=None,
                   help="ray near bound (auto-estimated from the scene if omitted)")
    p.add_argument("--far", type=float, default=None,
                   help="ray far bound (auto-estimated from the scene if omitted)")
    p.add_argument("--max_train_images", type=int, default=None)
    # Field / SH.
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--log2_hashmap_size", type=int, default=21)
    p.add_argument("--n_levels", type=int, default=16)
    p.add_argument("--base_resolution", type=int, default=16)
    p.add_argument("--max_resolution", type=int, default=1024)
    p.add_argument("--mlp_hidden_dim", type=int, default=64)
    p.add_argument("--mlp_hidden_layers", type=int, default=2)
    # Optimization.
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4096, help="rays per iteration")
    p.add_argument("--n_samples", type=int, default=256, help="samples per ray")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lr_final", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    # Logging / eval.
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_field(args, aabb) -> InstantNGPSHField:
    config = FieldConfig(
        sh_degree=args.sh_degree,
        n_levels=args.n_levels,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=args.base_resolution,
        max_resolution=args.max_resolution,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_num_hidden_layers=args.mlp_hidden_layers,
        predict_density=True,
    )
    return InstantNGPSHField(aabb=aabb, config=config)


@torch.no_grad()
def evaluate(field, dataset, args, device, out_dir, step, render_aabb=None, max_views=3):
    field.eval()
    psnrs = []
    n = min(max_views, dataset.num_images())
    for idx in range(n):
        rays_o, rays_d = dataset.rays_for_image(idx)
        out = render_image(
            field,
            rays_o.to(device),
            rays_d.to(device),
            near=dataset.near,
            far=dataset.far,
            n_samples=args.n_samples,
            sh_degree=args.sh_degree,
            bg_color=dataset.bg_color.to(device) if dataset.bg_color is not None else None,
            aabb=render_aabb,
        )
        gt = dataset.images[idx].to(device)
        mse = torch.mean((out["rgb"] - gt) ** 2).item()
        psnrs.append(mse_to_psnr(mse))

        rgb = (out["rgb"].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        imageio.imwrite(os.path.join(out_dir, f"eval_step{step:06d}_view{idx}.png"), rgb)
    field.train()
    return sum(psnrs) / len(psnrs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. tiny-cuda-nn requires a CUDA GPU; cannot train on CPU."
        )

    print(f"Loading training data from {args.data} ...")
    train_set = ColmapDataset.load(
        args.data,
        split="train",
        white_background=args.white_bg,
        downscale=args.downscale,
        max_images=args.max_train_images,
        near=args.near,
        far=args.far,
        images_dir=args.images_dir,
        holdout=args.holdout,
    ).to(device)
    print(f"  {train_set.num_images()} images at {train_set.H}x{train_set.W}, "
          f"focal={train_set.focal:.2f}, near={train_set.near:.4f}, far={train_set.far:.4f}")

    aabb = train_set.compute_aabb(padding=0.1).to(device)
    print(f"Scene AABB: {aabb.tolist()}")

    # Bound per-ray sampling by the scene box (COLMAP scenes can be large/unbounded).
    render_aabb = aabb

    field = build_field(args, aabb).to(device)
    n_params = sum(p.numel() for p in field.parameters())
    print(f"Field parameters: {n_params/1e6:.2f}M  "
          f"(per_level_scale={field.per_level_scale:.4f})")

    optimizer = torch.optim.Adam(
        field.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15
    )
    gamma = (args.lr_final / args.lr) ** (1.0 / max(args.iters, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    bg = train_set.bg_color.to(device) if train_set.bg_color is not None else None

    print(f"Training for {args.iters} iterations ...")
    t0 = time.time()
    running = 0.0
    for step in range(1, args.iters + 1):
        rays_o, rays_d, target_rgb = train_set.sample_rays(args.batch)

        out = volume_render_rays(
            field,
            rays_o,
            rays_d,
            near=train_set.near,
            far=train_set.far,
            n_samples=args.n_samples,
            sh_degree=args.sh_degree,
            bg_color=bg,
            perturb=True,
            aabb=render_aabb,
        )
        loss = torch.mean((out["rgb"] - target_rgb) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running += loss.item()
        if step % args.log_every == 0:
            avg = running / args.log_every
            running = 0.0
            rate = step / (time.time() - t0)
            print(f"[{step:6d}/{args.iters}] loss={avg:.5f} psnr={mse_to_psnr(avg):.2f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} ({rate:.1f} it/s)")

        if step % args.eval_every == 0 or step == args.iters:
            psnr = evaluate(field, train_set, args, device, args.out, step,
                            render_aabb=render_aabb)
            print(f"  [eval] train-view PSNR={psnr:.2f}")
            ckpt = os.path.join(args.out, "field.pt")
            torch.save(
                {
                    "state_dict": field.state_dict(),
                    "config": field.config.__dict__,
                    "aabb": field.aabb.cpu(),
                    "args": vars(args),
                    "step": step,
                },
                ckpt,
            )
    print(f"Done. Checkpoint saved to {os.path.join(args.out, 'field.pt')}")


if __name__ == "__main__":
    main()
