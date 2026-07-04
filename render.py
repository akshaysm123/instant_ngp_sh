"""Render a trained Instant-NGP field on a depth-range split and report metrics.

Renders each view by querying the field at its per-pixel surface points (from the
exported splat depth ranges) instead of volume rendering, and reports PSNR (plus SSIM
and LPIPS when ``--metrics`` is given, e.g. for held-out test-set evaluation).

    python -m instant_ngp_sh.render --data /path/to/garden \
        --depth_dir /path/to/output/garden/depth_ranges/ours_30000 \
        --ckpt runs/garden/field.pt --split test --metrics

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

from .depth_dataset import DepthRangeDataset
from .model import FieldConfig, InstantNGPField
from .train import render_view


def mse_to_psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def build_metrics(device):
    """Build SSIM + LPIPS metric modules on ``device``.

    Returns ``{"ssim": ..., "lpips": ...}``. LPIPS uses the AlexNet backbone and expects
    inputs in ``[0, 1]`` (``normalize=True``). Raises a helpful error if ``torchmetrics``
    is not installed.
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception as exc:  # pragma: no cover - depends on env
        raise SystemExit(
            f"--metrics needs torchmetrics ({exc}). Install with:\n"
            f"  pip install torchmetrics torchvision"
        )
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    return {"ssim": ssim, "lpips": lpips}


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained Instant-NGP field")
    p.add_argument("--data", required=True,
                   help="COLMAP scene dir with images/ and sparse/0/*.bin")
    p.add_argument("--depth_dir", required=True,
                   help="depth-range export dir (.../depth_ranges/ours_<iter>)")
    p.add_argument("--ckpt", required=True, help="path to field.pt checkpoint")
    p.add_argument("--out", default=None, help="output dir (default: alongside ckpt)")
    p.add_argument("--split", default="test", choices=["train", "test"],
                   help="which depth-range subfolder to read views from")
    p.add_argument("--holdout_every", type=int, default=0,
                   help="if >0, defines the held-out test set as every Nth view "
                        "(indices 0, N, 2N, ...); combine with --holdout_role")
    p.add_argument("--holdout_role", default="all", choices=["all", "train", "test"],
                   help="which side of the holdout split to render: 'train' (kept "
                        "views), 'test' (held-out views) or 'all' (default, everything)")
    p.add_argument("--images_dir", default="images", help="image subfolder")
    p.add_argument("--alpha_min", type=float, default=0.5)
    p.add_argument("--eval_chunk", type=int, default=1 << 18)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--metrics", action="store_true",
                   help="also compute SSIM and LPIPS (needs torchmetrics + torchvision); "
                        "used for held-out test-set evaluation")
    p.add_argument("--black_bg", action="store_true",
                   help="treat missing/background (no-surface) pixels as black and score "
                        "the full frame including them (PSNR over all pixels), instead of "
                        "masking them out of PSNR; forces a black background for the render "
                        "and the GT composite")
    p.add_argument("--keep_gt_bg", action="store_true",
                   help="with --black_bg, do NOT black out the ground truth: the render "
                        "still fills missing pixels with black, but the GT keeps its real "
                        "background, so the full-frame metrics penalize the unmodeled "
                        "background (a stricter, more comparable number). No effect "
                        "without --black_bg")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; tiny-cuda-nn requires a CUDA GPU.")

    # Read the checkpoint to CPU first (config/args/weights), but do NOT build the field
    # yet. Initializing CUDA/tcnn allocates ~1.5-2 GB of *host* context that would sit
    # resident during the dataset load and push the peak past a memory-capped host (e.g.
    # WSL). Mirror train.py: load the (multi-GB) dataset first, then bring up CUDA.
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = FieldConfig(**ckpt["config"])
    train_args = ckpt.get("args", {})
    white_bg = train_args.get("white_bg", False)

    dataset = DepthRangeDataset.load(
        args.data,
        depth_dir=args.depth_dir,
        split=args.split,
        images_dir=args.images_dir,
        alpha_min=args.alpha_min,
        white_background=white_bg,
        max_images=args.max_images,
        holdout_every=args.holdout_every,
        holdout_role=args.holdout_role,
    ).to(device, images_on_device=False)

    # Now bring up CUDA/tcnn and load the trained weights, then free the CPU checkpoint.
    field = InstantNGPField(aabb=ckpt["aabb"], config=config).to(device)
    field.load_state_dict(ckpt["state_dict"])
    field.eval()
    del ckpt

    default_name = f"render_{args.split}"
    if args.holdout_every > 0 and args.holdout_role != "all":
        default_name = f"render_{args.holdout_role}"
    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt), default_name)
    os.makedirs(out_dir, exist_ok=True)

    metrics = build_metrics(device) if args.metrics else None
    # --black_bg forces a black background for both the render fill and the GT composite,
    # and scores the full frame (missing pixels count as black-vs-black).
    if args.black_bg and dataset.bg_color is not None:
        dataset.bg_color = torch.zeros_like(dataset.bg_color)
    bg = (dataset.bg_color.to(device) if dataset.bg_color is not None
          else torch.zeros(3, device=device))

    psnrs, ssims, lpipss = [], [], []
    for idx in range(dataset.num_images()):
        img = render_view(field, dataset, idx, device, args.eval_chunk)
        gt = dataset.image(idx).to(device)
        valid = dataset.valid[idx].to(device)
        pred = img.clamp(0, 1)
        # Ground truth used for the spatial (SSIM/LPIPS) and full-frame (--black_bg)
        # scores. Normally the GT is composited onto the same background as the render on
        # no-surface pixels so those regions match exactly. With --black_bg --keep_gt_bg
        # we instead keep the real GT background, so the render's black fill is penalized
        # against the true (unmodeled) background rather than getting free matching pixels.
        if args.black_bg and args.keep_gt_bg:
            gt_comp = gt
        else:
            gt_comp = torch.where(valid[..., None], gt, bg)

        psnr = ssim = lpips = float("nan")
        if args.black_bg:
            mse = torch.mean((pred - gt_comp) ** 2).item()
            psnr = mse_to_psnr(mse)
            psnrs.append(psnr)
            scored = True
        elif valid.any():
            mse = torch.mean((pred[valid] - gt[valid]) ** 2).item()
            psnr = mse_to_psnr(mse)
            psnrs.append(psnr)
            scored = True
        else:
            scored = False

        if metrics is not None and scored:
            p_ = pred.permute(2, 0, 1)[None].contiguous()
            t_ = gt_comp.permute(2, 0, 1)[None].contiguous()
            ssim = float(metrics["ssim"](p_, t_))
            lpips = float(metrics["lpips"](p_, t_))
            metrics["ssim"].reset()
            metrics["lpips"].reset()
            ssims.append(ssim)
            lpipss.append(lpips)

        rgb = (pred.cpu().numpy() * 255).astype("uint8")
        name = dataset.image_names[idx]
        imageio.imwrite(os.path.join(out_dir, f"{name}.png"), rgb)
        if metrics is not None:
            print(f"  view {idx:03d} ({name}): PSNR={psnr:.2f} "
                  f"SSIM={ssim:.4f} LPIPS={lpips:.4f}")
        else:
            print(f"  view {idx:03d} ({name}): PSNR={psnr:.2f}")

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\nmean over {len(psnrs)} views: PSNR={_mean(psnrs):.2f}", end="")
    if metrics is not None:
        print(f" SSIM={_mean(ssims):.4f} LPIPS={_mean(lpipss):.4f}", end="")
    print(f"\nRenders written to {out_dir}")


if __name__ == "__main__":
    main()
