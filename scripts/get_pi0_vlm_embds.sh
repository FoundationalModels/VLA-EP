#!/usr/bin/env bash
# Extract Pi0 VLM embeddings for every task and instruction variant in ref_exp.json.
#
# Always extracts base pretrained PaliGemma embeddings.
# Pass --pi0_pretrained_path to also extract bridge-pretrained Pi0 embeddings.
# Pass --checkpoint_path to also extract task-specific post-trained Pi0 embeddings.
#
# PaliGemma weights are downloaded automatically from HuggingFace if not present.
# Default location: $TRANSFORMERS_CACHE/paligemma-3b-pt-224 (or
# ~/.cache/huggingface/paligemma-3b-pt-224 when TRANSFORMERS_CACHE is unset).
#
# Output files in OUTPUT_DIR:
#   {task}_{paligemma_model}_pretrained_embds.npz
#   {task}_{pi0_pretrained_stem}_pi0pretrained_embds.npz  (only when --pi0_pretrained_path given)
#   {task}_{checkpoint_stem}_finetuned_embds.npz          (only when --checkpoint_path given)
#
# Usage:
#   # PaliGemma base only
#   ./get_pi0_vlm_embds.sh
#
#   # all three weight sets
#   ./get_pi0_vlm_embds.sh \
#       --pi0_pretrained_path ./checkpoints/pi0/bridge_beta_step19296_2024-12-26_22-30_42.pt \
#       --checkpoint_path ./checkpoints/pi0_bridge+sink_carrot-knife_100000_noaug.pt
#
#   # all options
#   ./get_pi0_vlm_embds.sh --pretrained_model_path /path/to/paligemma-3b-pt-224 \
#                          --pi0_pretrained_path /path/to/bridge_pretrained.pt \
#                          --checkpoint_path /path/to/task_checkpoint.pt \
#                          --output_dir ./embeddings --device cuda --use_bf16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PI_ZERO_DIR="$REPO_DIR/open-pi-zero"
VENV="$PI_ZERO_DIR/.venv"
PY_SCRIPT="$PI_ZERO_DIR/scripts/get_pi0_vlm_embds.py"

REF_EXP_PATH="$REPO_DIR/media/ref_exp.jsonc"
IMAGE_BASE_DIR="$REPO_DIR/media"
OUTPUT_DIR="$REPO_DIR/embeddings/pi0"
PRETRAINED_MODEL_PATH=""
PI0_PRETRAINED_PATH=""
CHECKPOINT_PATH=""
TASKS_ARG=""
DEVICE="cuda"
USE_BF16_FLAG=""

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained_model_path) PRETRAINED_MODEL_PATH="$2";   shift 2 ;;
        --pi0_pretrained_path)   PI0_PRETRAINED_PATH="$2";     shift 2 ;;
        --checkpoint_path)       CHECKPOINT_PATH="$2";         shift 2 ;;
        --tasks)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TASKS_ARG="$TASKS_ARG $1"; shift
            done ;;
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

if [[ -n "$PI0_PRETRAINED_PATH" && ! -f "$PI0_PRETRAINED_PATH" ]]; then
    echo "Error: pi0 pretrained checkpoint not found: $PI0_PRETRAINED_PATH"
    exit 1
fi

if [[ -n "$CHECKPOINT_PATH" && ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: task checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "============================================"
echo "Pi0 VLM embedding extraction"
echo "  pretrained model    : ${PRETRAINED_MODEL_PATH:-<default/auto-download>}"
echo "  pi0 pretrained (.pt): ${PI0_PRETRAINED_PATH:-none}"
echo "  task checkpoint(.pt): ${CHECKPOINT_PATH:-none}"
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

PRETRAINED_ARG=""
if [[ -n "$PRETRAINED_MODEL_PATH" ]]; then
    PRETRAINED_ARG="--pretrained_model_path $PRETRAINED_MODEL_PATH"
fi

PI0_PRETRAINED_ARG=""
if [[ -n "$PI0_PRETRAINED_PATH" ]]; then
    PI0_PRETRAINED_PATH="$(realpath "$PI0_PRETRAINED_PATH")"
    PI0_PRETRAINED_ARG="--pi0_pretrained_path $PI0_PRETRAINED_PATH"
fi

CHECKPOINT_ARG=""
if [[ -n "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH="$(realpath "$CHECKPOINT_PATH")"
    CHECKPOINT_ARG="--checkpoint_path $CHECKPOINT_PATH"
fi

# Run from open-pi-zero/ so that config/ and src/ relative imports work
cd "$PI_ZERO_DIR"

TASKS_FLAG=""
if [[ -n "$TASKS_ARG" ]]; then
    TASKS_FLAG="--tasks $TASKS_ARG"
fi

python "$PY_SCRIPT" \
    --ref_exp_path      "$REF_EXP_PATH" \
    --image_base_dir    "$IMAGE_BASE_DIR" \
    --output_dir        "$OUTPUT_DIR" \
    --config_path       "config/train/bridge.yaml" \
    --device            "$DEVICE" \
    $PRETRAINED_ARG \
    $PI0_PRETRAINED_ARG \
    $CHECKPOINT_ARG \
    $TASKS_FLAG \
    $USE_BF16_FLAG
