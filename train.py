"""Train the Instant-NGP field by *direct point-color supervision*.

The geometry comes from a converged 2D Gaussian-splatting model: for every pixel of
every training view we have a depth segment (front/back/median surface depth) where
splats exist (see ``notes/data.md``). We back-project jittered points along those segments
to world space, read the ground-truth pixel color, and fit the field as a plain regression

    rgb = field(world_point, view_dir)   ->   MSE against the ground-truth color

No volume rendering and no density are involved: the field is purely a
position+direction -> RGB texture atlas that can later color the splatting. The scene AABB
is the box containing the splat surface (back-projected segment endpoints).

Run as a module (recommended)::

    python -m instant_ngp_sh.train --data /path/to/garden \
        --depth_dir /path/to/output/garden/depth_ranges/ours_30000 \
        --out runs/garden --images_dir images_4

Requires a CUDA GPU and tiny-cuda-nn.
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

from .depth_dataset import DepthRangeDataset
from .model import FieldConfig, InstantNGPField


def mse_to_psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train an Instant-NGP field by direct point-color supervision"
    )
    # Data.
    p.add_argument("--data", required=True,
                   help="COLMAP scene dir with images/ and sparse/0/{cameras,images}.bin")
    p.add_argument("--depth_dir", required=True,
                   help="depth-range export dir (.../depth_ranges/ours_<iter>) with "
                        "train/ and test/ subfolders of per-view .npz files")
    p.add_argument("--out", default="runs/ngp_sh", help="output directory")
    p.add_argument("--images_dir", default="images",
                   help="ground-truth image subfolder, e.g. images_4")
    p.add_argument("--split", default="train", choices=["train", "test"],
                   help="which depth-range split to train on")
    p.add_argument("--holdout_every", type=int, default=0,
                   help="if >0, hold out every Nth view (indices 0, N, 2N, ...) as a "
                        "test set and train only on the remaining views; 0 (default) "
                        "trains on every view")
    p.add_argument("--alpha_min", type=float, default=0.5,
                   help="min accumulated opacity for a pixel to count as a real surface")
    p.add_argument("--white_bg", action="store_true", default=False,
                   help="composite empty space onto white during eval (default black)")
    p.add_argument("--max_train_images", type=int, default=None)
    # Field.
    p.add_argument("--sh_degree", type=int, default=4,
                   help="degree of the SH view-direction encoding (tcnn: degree**2 features)")
    p.add_argument("--log2_hashmap_size", type=int, default=21)
    p.add_argument("--n_levels", type=int, default=16)
    p.add_argument("--base_resolution", type=int, default=16)
    p.add_argument("--max_resolution", type=int, default=1024)
    p.add_argument("--geo_feat_dim", type=int, default=15)
    p.add_argument("--mlp_hidden_dim", type=int, default=64)
    p.add_argument("--mlp_hidden_layers", type=int, default=1)
    p.add_argument("--color_mlp_hidden_dim", type=int, default=64)
    p.add_argument("--color_mlp_hidden_layers", type=int, default=2)
    # Optimization.
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--batch", type=int, default=16384, help="points per iteration")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lr_final", type=float, default=1e-4)
    p.add_argument("--width_tau", type=float, default=0.0,
                   help="if >0, weight each sample by exp(-width/tau) to down-weight wide "
                        "(edge/transparency) depth segments; in world depth units")
    p.add_argument("--aabb_padding", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    # Logging / eval.
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_chunk", type=int, default=1 << 18,
                   help="points per chunk during eval rendering (lower if eval OOMs)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--images_on_gpu", action="store_true",
                   help="store all training images/depths on GPU (default: keep on CPU)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_field(args, aabb) -> InstantNGPField:
    config = FieldConfig(
        sh_degree=args.sh_degree,
        n_levels=args.n_levels,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=args.base_resolution,
        max_resolution=args.max_resolution,
        geo_feat_dim=args.geo_feat_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_num_hidden_layers=args.mlp_hidden_layers,
        color_mlp_hidden_dim=args.color_mlp_hidden_dim,
        color_mlp_num_hidden_layers=args.color_mlp_hidden_layers,
    )
    return InstantNGPField(aabb=aabb, config=config)


@torch.no_grad()
def render_view(field, dataset, idx, device, chunk):
    """Render view ``idx`` by querying the field at its per-pixel surface points."""
    points, dirs, valid = dataset.surface_points_for_image(idx)
    H, W = dataset.H, dataset.W
    img = torch.zeros(H, W, 3, device=device)
    if dataset.bg_color is not None:
        img += dataset.bg_color.to(device)

    pts = points[valid]
    dvs = dirs[valid]
    if pts.numel() > 0:
        out = torch.empty(pts.shape[0], 3, device=device)
        for i in range(0, pts.shape[0], chunk):
            j = min(i + chunk, pts.shape[0])
            out[i:j] = field(pts[i:j], dvs[i:j])
        img[valid] = out
    return img


@torch.no_grad()
def evaluate(field, dataset, args, device, out_dir, step, max_views=3):
    field.eval()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    psnrs = []
    n = min(max_views, dataset.num_images())
    for idx in range(n):
        img = render_view(field, dataset, idx, device, args.eval_chunk)
        gt = dataset.image(idx).to(device)
        # Only score pixels that have a surface (others are background fill).
        valid = dataset.valid[idx].to(device)
        if valid.any():
            mse = torch.mean((img[valid] - gt[valid]) ** 2).item()
            psnrs.append(mse_to_psnr(mse))

        rgb = (img.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        imageio.imwrite(os.path.join(out_dir, f"eval_step{step:06d}_view{idx}.png"), rgb)
        del img
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    field.train()
    return sum(psnrs) / len(psnrs) if psnrs else float("nan")


def main():
    args = parse_args()
    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. tiny-cuda-nn requires a CUDA GPU; cannot train on CPU."
        )

    holdout_role = "train" if args.holdout_every > 0 else "all"
    print(f"Loading depth-range data from {args.depth_dir} ({args.split}) ...")
    if args.holdout_every > 0:
        print(f"  holding out every {args.holdout_every}th view as a test set "
              f"(training on the rest)")
    train_set = DepthRangeDataset.load(
        args.data,
        depth_dir=args.depth_dir,
        split=args.split,
        images_dir=args.images_dir,
        alpha_min=args.alpha_min,
        white_background=args.white_bg,
        max_images=args.max_train_images,
        holdout_every=args.holdout_every,
        holdout_role=holdout_role,
    ).to(device, images_on_device=args.images_on_gpu)
    n_valid = train_set.num_valid
    print(f"  {train_set.num_images()} views at {train_set.H}x{train_set.W}, "
          f"{n_valid/1e6:.1f}M valid surface pixels")

    aabb = train_set.compute_aabb(padding=args.aabb_padding).to(device)
    print(f"Splat AABB: {aabb.tolist()}")

    field = build_field(args, aabb).to(device)
    n_params = sum(p.numel() for p in field.parameters())
    print(f"Field parameters: {n_params/1e6:.2f}M  "
          f"(per_level_scale={field.per_level_scale:.4f})")

    optimizer = torch.optim.Adam(
        field.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-15
    )
    gamma = (args.lr_final / args.lr) ** (1.0 / max(args.iters, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    print(f"Training for {args.iters} iterations ...")
    t0 = time.time()
    running = 0.0
    for step in range(1, args.iters + 1):
        points, dirs, target_rgb, widths = train_set.sample_points(args.batch)

        rgb = field(points, dirs)
        sq_err = (rgb - target_rgb) ** 2
        if args.width_tau > 0:
            w = torch.exp(-widths / args.width_tau)[:, None]
            loss = (w * sq_err).sum() / (w.sum() * 3.0 + 1e-12)
        else:
            loss = sq_err.mean()

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
            psnr = evaluate(field, train_set, args, device, args.out, step)
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
