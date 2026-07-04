#!/usr/bin/env bash
#
# Train the Instant-NGP appearance field on a scene's exported depth ranges, then
# render every view at its expected (median) surface depth -- the same query used by
# the notebook's `render_view`. Renders are split into train/ and test/ subfolders.
#
# Runs locally (plain bash + the current Python env; needs a CUDA GPU + tiny-cuda-nn).
#
# Usage:
#   ./train_and_render.sh --data <colmap_dir> --depth_dir <depth_ranges_dir> [options]
#
# The COLMAP scene (images/ + sparse/) and the depth-range NPZs may live in completely
# different folders; pass each with its own flag. If --depth_dir is omitted it defaults
# to <data>/depth_ranges.
#
# Held-out test views:
#   By default every view is used for training and rendered into renders/train/.
#   Pass --holdout_every N to hold out every Nth view (indices 0, N, 2N, ...) as a test
#   set: training then uses only the remaining views, and the held-out views are rendered
#   into renders/test/ (the rest into renders/train/).
#
# Examples:
#   # Train on everything, render all views into renders/train/
#   ./train_and_render.sh --data data/counter --depth_dir data/counter/depth_ranges --out out/counter --images_dir images_4
#
#   # Hold out every 8th view as a test split
#   ./train_and_render.sh --data data/counter --depth_dir data/counter/depth_ranges \
#       --out out/counter --images_dir images_4 --holdout_every 8

set -euo pipefail

# --- resolve repo location so `python -m instant_ngp_sh.*` always works ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../instant_ngp_sh
REPO_PARENT="$(dirname "$SCRIPT_DIR")"                       # parent that holds the pkg
PKG_NAME="$(basename "$SCRIPT_DIR")"                         # usually "instant_ngp_sh"

# --- defaults (override via flags) ----------------------------------------------------
DATA=""
DEPTH_DIR=""
OUT=""
IMAGES_DIR="images_4"
SPLIT="train"           # depth-range subfolder to read views from
HOLDOUT_EVERY=0         # 0 => train on everything (no test split)
ITERS=4000
BATCH=65536
SH_DEGREE=4
DEVICE="cuda"
BLACK_BG=0              # 1 => score test metrics over the full frame with black background
KEEP_GT_BG=0            # 1 => (with --black_bg) keep the real GT background instead of blacking it
EXTRA_ARGS=()           # anything after `--` is forwarded verbatim to train.py

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)          DATA="$2"; shift 2 ;;
        --depth_dir)     DEPTH_DIR="$2"; shift 2 ;;
        --out)           OUT="$2"; shift 2 ;;
        --images_dir)    IMAGES_DIR="$2"; shift 2 ;;
        --split)         SPLIT="$2"; shift 2 ;;
        --holdout_every) HOLDOUT_EVERY="$2"; shift 2 ;;
        --iters)         ITERS="$2"; shift 2 ;;
        --batch)         BATCH="$2"; shift 2 ;;
        --sh_degree)     SH_DEGREE="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --black_bg)      BLACK_BG=1; shift ;;
        --keep_gt_bg)    KEEP_GT_BG=1; shift ;;
        -h|--help)       usage 0 ;;
        --)              shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

# --- validation -----------------------------------------------------------------------
if [[ -z "$DATA" ]]; then
    echo "Error: --data <colmap_scene_dir> is required." >&2
    usage 1
fi
DATA="$(cd "$DATA" && pwd)"                       # absolute, and check it exists
DEPTH_DIR="${DEPTH_DIR:-$DATA/depth_ranges}"
if [[ ! -d "$DEPTH_DIR" ]]; then
    echo "Error: depth_dir not found: $DEPTH_DIR" >&2
    exit 1
fi
DEPTH_DIR="$(cd "$DEPTH_DIR" && pwd)"
OUT="${OUT:-$SCRIPT_DIR/out/$(basename "$DATA")}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
CKPT="$OUT/field.pt"
RENDER_DIR="$OUT/renders"

export PYTHONUNBUFFERED=1
# Cap glibc's per-thread malloc arenas so freed transient buffers (npz decompression,
# dtype casts, image resizes during dataset load) are not retained across threads and
# inflate RSS on memory-capped hosts like WSL.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
cd "$REPO_PARENT"

echo "======================================================================"
echo " data        : $DATA"
echo " depth_dir   : $DEPTH_DIR"
echo " images_dir  : $IMAGES_DIR"
echo " split       : $SPLIT"
echo " out         : $OUT"
echo " holdout     : $([[ "$HOLDOUT_EVERY" -gt 0 ]] && echo "every ${HOLDOUT_EVERY}th view -> test" || echo "none (train on all views)")"
echo " iters/batch : $ITERS / $BATCH"
echo "======================================================================"

# --- 1. train -------------------------------------------------------------------------
echo ""
echo "=== [1/3] Training ==="
python -u -m "$PKG_NAME.train" \
    --data "$DATA" \
    --depth_dir "$DEPTH_DIR" \
    --out "$OUT" \
    --images_dir "$IMAGES_DIR" \
    --split "$SPLIT" \
    --holdout_every "$HOLDOUT_EVERY" \
    --iters "$ITERS" \
    --batch "$BATCH" \
    --sh_degree "$SH_DEGREE" \
    --device "$DEVICE" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# --- 2. render the training views (expected-depth query, like the notebook) -----------
echo ""
echo "=== [2/3] Rendering train views -> $RENDER_DIR/train ==="
python -u -m "$PKG_NAME.render" \
    --data "$DATA" \
    --depth_dir "$DEPTH_DIR" \
    --ckpt "$CKPT" \
    --out "$RENDER_DIR/train" \
    --split "$SPLIT" \
    --images_dir "$IMAGES_DIR" \
    --holdout_every "$HOLDOUT_EVERY" \
    --holdout_role "$([[ "$HOLDOUT_EVERY" -gt 0 ]] && echo train || echo all)" \
    --device "$DEVICE"

# --- 3. render the held-out test views (only when a holdout was requested) -------------
if [[ "$HOLDOUT_EVERY" -gt 0 ]]; then
    echo ""
    echo "=== [3/3] Rendering held-out test views -> $RENDER_DIR/test ==="
    python -u -m "$PKG_NAME.render" \
        --data "$DATA" \
        --depth_dir "$DEPTH_DIR" \
        --ckpt "$CKPT" \
        --out "$RENDER_DIR/test" \
        --split "$SPLIT" \
        --images_dir "$IMAGES_DIR" \
        --holdout_every "$HOLDOUT_EVERY" \
        --holdout_role test \
        --metrics \
        $([[ "$BLACK_BG" -eq 1 ]] && echo --black_bg) \
        $([[ "$KEEP_GT_BG" -eq 1 ]] && echo --keep_gt_bg) \
        --device "$DEVICE"
else
    echo ""
    echo "=== [3/3] No holdout requested (--holdout_every 0); skipping test renders ==="
fi

echo ""
echo "Done. Checkpoint: $CKPT"
echo "Renders: $RENDER_DIR/{train,test}"
