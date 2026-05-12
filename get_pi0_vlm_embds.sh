#!/usr/bin/env bash
# Extract Pi0 VLM embeddings for every task and instruction variant in ref_exp.json.
#
# Always extracts base pretrained PaliGemma embeddings.
# Pass --checkpoint_path to also extract post-trained (Bridge fine-tuned) embeddings.
#
# PaliGemma weights are downloaded automatically from HuggingFace if not present.
# Default location: $TRANSFORMERS_CACHE/paligemma-3b-pt-224 (or
# ~/.cache/huggingface/paligemma-3b-pt-224 when TRANSFORMERS_CACHE is unset).
#
# Output files in OUTPUT_DIR:
#   {task}_{paligemma_model}_pretrained_embds.npz
#   {task}_{checkpoint_stem}_finetuned_embds.npz   (only when --checkpoint_path given)
#
# Usage:
#   # pretrained only — weights downloaded automatically if needed
#   ./get_pi0_vlm_embds.sh
#
#   # pretrained + finetuned
#   ./get_pi0_vlm_embds.sh --checkpoint_path /path/to/checkpoint.pt
#
#   # all options
#   ./get_pi0_vlm_embds.sh --pretrained_model_path /path/to/paligemma-3b-pt-224 \
#                          --checkpoint_path /path/to/checkpoint.pt \
#                          --output_dir ./embeddings --device cuda --use_bf16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_ZERO_DIR="$SCRIPT_DIR/open-pi-zero"
VENV="$PI_ZERO_DIR/.venv"
PY_SCRIPT="$PI_ZERO_DIR/scripts/get_pi0_vlm_embds.py"

REF_EXP_PATH="$SCRIPT_DIR/media/ref_exp.json"
IMAGE_BASE_DIR="$SCRIPT_DIR/media"
OUTPUT_DIR="$SCRIPT_DIR/embeddings"
PRETRAINED_MODEL_PATH=""
CHECKPOINT_PATH=""
DEVICE="cuda"
USE_BF16_FLAG=""

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained_model_path) PRETRAINED_MODEL_PATH="$2"; shift 2 ;;
        --checkpoint_path)       CHECKPOINT_PATH="$2";       shift 2 ;;
        --output_dir)            OUTPUT_DIR="$2";            shift 2 ;;
        --device)                DEVICE="$2";                shift 2 ;;
        --use_bf16)              USE_BF16_FLAG="--use_bf16"; shift   ;;
        -h|--help)               usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ! -f "$REF_EXP_PATH" ]]; then
    echo "Error: ref_exp.json not found at $REF_EXP_PATH"
    exit 1
fi

if [[ -n "$CHECKPOINT_PATH" && ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "============================================"
echo "Pi0 VLM embedding extraction"
echo "  pretrained model : ${PRETRAINED_MODEL_PATH:-<default/auto-download>}"
echo "  checkpoint (.pt) : ${CHECKPOINT_PATH:-none (pretrained only)}"
echo "  ref_exp          : $REF_EXP_PATH"
echo "  images from      : $IMAGE_BASE_DIR"
echo "  output dir       : $OUTPUT_DIR"
echo "  device           : $DEVICE"
echo "  bf16             : ${USE_BF16_FLAG:-no}"
echo "============================================"

# Activate virtual environment
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "Warning: venv not found at $VENV — using system Python"
fi

# Ensure libcuda.so is on the library path (needed when the shell env is minimal)
export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Run from open-pi-zero/ so that config/ and src/ relative imports work
cd "$PI_ZERO_DIR"

PRETRAINED_ARG=""
if [[ -n "$PRETRAINED_MODEL_PATH" ]]; then
    PRETRAINED_ARG="--pretrained_model_path $PRETRAINED_MODEL_PATH"
fi

CHECKPOINT_ARG=""
if [[ -n "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_ARG="--checkpoint_path $CHECKPOINT_PATH"
fi

python "$PY_SCRIPT" \
    --ref_exp_path  "$REF_EXP_PATH" \
    --image_base_dir "$IMAGE_BASE_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --config_path   "config/train/bridge.yaml" \
    --device        "$DEVICE" \
    $PRETRAINED_ARG \
    $CHECKPOINT_ARG \
    $USE_BF16_FLAG
