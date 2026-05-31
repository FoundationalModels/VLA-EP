#!/usr/bin/env bash
# Extract Pi0 VLM embeddings for all phases and all tasks.
#
# Runs extraction in two groups (split by co-trained checkpoint):
#   Group 1: put_carrot_on_plate + put_knife_on_plate  (carrot/knife checkpoint)
#   Group 2: flip_pot_upright + put_plate_in_sink       (pot/plate checkpoint)
#
# Each group extracts all three phases:
#   1. Base PaliGemma weights
#   2. Pi0 bridge-pretrained checkpoint
#   3. Task-group co-trained checkpoint
#
# Usage:
#   ./get_pi0_vlm_embds.sh
#
#   # override checkpoint paths
#   ./get_pi0_vlm_embds.sh \
#       --carrot_knife_checkpoint ./checkpoints/pi0_cotrained_bridge+sink_carrot-knife_100000_noaug.pt \
#       --pot_plate_checkpoint    ./checkpoints/pi0_cotrained_bridge+sink_pot-plate_100000_noaug.pt
#
#   # all options
#   ./get_pi0_vlm_embds.sh \
#       --pretrained_model_path ./checkpoints/pi0_basevlm_paligemma-3b-pt-224 \
#       --pi0_pretrained_path   ./checkpoints/pi0_pretrained_bridge_beta_step19296_2024-12-26_22-30_42.pt \
#       --carrot_knife_checkpoint ./checkpoints/pi0_cotrained_bridge+sink_carrot-knife_100000_noaug.pt \
#       --pot_plate_checkpoint    ./checkpoints/pi0_cotrained_bridge+sink_pot-plate_100000_noaug.pt \
#       --output_dir ./embeddings/pi0 --device cuda --use_bf16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_ZERO_DIR="$REPO_DIR/open-pi-zero"
VENV="$PI_ZERO_DIR/.venv"
PY_SCRIPT="$PI_ZERO_DIR/scripts/get_pi0_vlm_embds.py"

REF_EXP_PATH="$REPO_DIR/media/ref_exp.jsonc"
IMAGE_BASE_DIR="$REPO_DIR/media"
OUTPUT_DIR="$REPO_DIR/embeddings/pi0"
PRETRAINED_MODEL_PATH="$REPO_DIR/checkpoints/pi0_basevlm_paligemma-3b-pt-224"
PI0_PRETRAINED_PATH="$REPO_DIR/checkpoints/pi0_pretrained_bridge_beta_step19296_2024-12-26_22-30_42.pt"
CARROT_KNIFE_CHECKPOINT="$REPO_DIR/checkpoints/pi0_cotrained_bridge+sink_carrot-knife_100000_noaug.pt"
POT_PLATE_CHECKPOINT="$REPO_DIR/checkpoints/pi0_cotrained_bridge+sink_pot-plate_100000_noaug.pt"
DEVICE="cuda"
USE_BF16_FLAG=""

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained_model_path)   PRETRAINED_MODEL_PATH="$2"; shift 2 ;;
        --pi0_pretrained_path)     PI0_PRETRAINED_PATH="$2";   shift 2 ;;
        --carrot_knife_checkpoint) CARROT_KNIFE_CHECKPOINT="$2"; shift 2 ;;
        --pot_plate_checkpoint)    POT_PLATE_CHECKPOINT="$2";   shift 2 ;;
        --output_dir)              OUTPUT_DIR="$2";             shift 2 ;;
        --device)                  DEVICE="$2";                 shift 2 ;;
        --use_bf16)                USE_BF16_FLAG="--use_bf16";  shift   ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ! -f "$REF_EXP_PATH" ]]; then
    echo "Error: ref_exp.jsonc not found at $REF_EXP_PATH"
    exit 1
fi

for ckpt in "$PI0_PRETRAINED_PATH" "$CARROT_KNIFE_CHECKPOINT" "$POT_PLATE_CHECKPOINT"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "Error: checkpoint not found: $ckpt"
        exit 1
    fi
done

PI0_PRETRAINED_PATH="$(realpath "$PI0_PRETRAINED_PATH")"
CARROT_KNIFE_CHECKPOINT="$(realpath "$CARROT_KNIFE_CHECKPOINT")"
POT_PLATE_CHECKPOINT="$(realpath "$POT_PLATE_CHECKPOINT")"

echo "============================================"
echo "Pi0 VLM embedding extraction"
echo "  pretrained model        : $PRETRAINED_MODEL_PATH"
echo "  pi0 pretrained (.pt)    : $PI0_PRETRAINED_PATH"
echo "  carrot/knife co-trained : $CARROT_KNIFE_CHECKPOINT"
echo "  pot/plate co-trained    : $POT_PLATE_CHECKPOINT"
echo "  ref_exp                 : $REF_EXP_PATH"
echo "  images from             : $IMAGE_BASE_DIR"
echo "  output dir              : $OUTPUT_DIR"
echo "  device                  : $DEVICE"
echo "  bf16                    : ${USE_BF16_FLAG:-no}"
echo "============================================"

if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "Warning: venv not found at $VENV — using system Python"
fi

export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$PI_ZERO_DIR"

COMMON_ARGS=(
    --ref_exp_path          "$REF_EXP_PATH"
    --image_base_dir        "$IMAGE_BASE_DIR"
    --output_dir            "$OUTPUT_DIR"
    --config_path           "config/train/bridge.yaml"
    --pretrained_model_path "$PRETRAINED_MODEL_PATH"
    --pi0_pretrained_path   "$PI0_PRETRAINED_PATH"
    --device                "$DEVICE"
)
if [[ -n "$USE_BF16_FLAG" ]]; then
    COMMON_ARGS+=("$USE_BF16_FLAG")
fi

echo ""
echo "========================================"
echo "Group 1: put_carrot_on_plate + put_knife_on_plate"
echo "========================================"
python "$PY_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --tasks put_carrot_on_plate put_knife_on_plate \
    --checkpoint_path "$CARROT_KNIFE_CHECKPOINT"

echo ""
echo "========================================"
echo "Group 2: flip_pot_upright + put_plate_in_sink"
echo "========================================"
python "$PY_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --tasks flip_pot_upright put_plate_in_sink \
    --checkpoint_path "$POT_PLATE_CHECKPOINT"

echo ""
echo "Done. Embeddings saved under $OUTPUT_DIR"
