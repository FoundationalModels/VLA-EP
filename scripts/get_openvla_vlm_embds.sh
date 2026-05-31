#!/usr/bin/env bash
# Extract OpenVLA VLM embeddings for every task and instruction variant in ref_exp.jsonc.
#
# Always extracts:
#   - Base Prismatic VLM embeddings (pre-robot-training, from TRI-ML/prismatic-vlms)
#   - OpenVLA pretrained embeddings (trained on Open-X, default: openvla/openvla-7b)
# Both mean-pooled and final-token embeddings are extracted in a single forward pass.
#
# Pass --checkpoint_path to also extract fine-tuned checkpoint embeddings.
#
# Output .npz keys per task:
#   {inst_key}_layer{idx}_mean    — mean-pooled hidden state
#   {inst_key}_layer{idx}_final   — final-token hidden state
#
# Usage:
#   # pretrained + base VLM only
#   ./get_openvla_vlm_embds.sh
#
#   # pretrained + base VLM + finetuned checkpoint
#   ./get_openvla_vlm_embds.sh --checkpoint_path /path/to/finetuned_checkpoint
#
#   # all options
#   ./get_openvla_vlm_embds.sh \
#       --base_vlm_path prism-dinosiglip-224px+7b \
#       --pretrained_model_path openvla/openvla-7b \
#       --checkpoint_path /path/to/finetuned_checkpoint \
#       --hf_token /path/to/.hf_token

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENVLA_PROBE_DIR="$REPO_DIR/openvla-probe"
CONDA_ENV="openvla-probe"
PY_SCRIPT="$OPENVLA_PROBE_DIR/scripts/get_openvla_vlm_embeds.py"

REF_EXP_PATH="$REPO_DIR/media/ref_exp.jsonc"
IMAGE_BASE_DIR="$REPO_DIR/media"
OUTPUT_DIR="$REPO_DIR/embeddings/openvla"
BASE_VLM_PATH="prism-dinosiglip-224px+7b"
PRETRAINED_MODEL_PATH="openvla/openvla-7b"
CHECKPOINT_PATH=""
HF_TOKEN_PATH=""
TASKS_ARG=""

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_vlm_path)        BASE_VLM_PATH="$2";        shift 2 ;;
        --pretrained_model_path) PRETRAINED_MODEL_PATH="$2"; shift 2 ;;
        --checkpoint_path)      CHECKPOINT_PATH="$2";      shift 2 ;;
        --hf_token)             HF_TOKEN_PATH="$2";        shift 2 ;;
        --output_dir)           OUTPUT_DIR="$2";           shift 2 ;;
        --tasks)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TASKS_ARG="$TASKS_ARG $1"; shift
            done ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ! -f "$REF_EXP_PATH" ]]; then
    echo "Error: ref_exp.jsonc not found at $REF_EXP_PATH"
    exit 1
fi

if [[ -n "$CHECKPOINT_PATH" && ! -d "$CHECKPOINT_PATH" && ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: checkpoint not found (expected a .pt file or safetensors directory): $CHECKPOINT_PATH"
    exit 1
fi

echo "============================================"
echo "OpenVLA VLM embedding extraction"
echo "  base VLM         : $BASE_VLM_PATH"
echo "  pretrained model : $PRETRAINED_MODEL_PATH"
echo "  checkpoint       : ${CHECKPOINT_PATH:-none (pretrained only)}"
echo "  ref_exp          : $REF_EXP_PATH"
echo "  images from      : $IMAGE_BASE_DIR"
echo "  output dir       : $OUTPUT_DIR"
echo "  pooling          : mean + final"
echo "============================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CHECKPOINT_ARG=""
if [[ -n "$CHECKPOINT_PATH" ]]; then
    # Resolve to absolute path before cd'ing into OPENVLA_PROBE_DIR
    CHECKPOINT_PATH="$(realpath "$CHECKPOINT_PATH")"
    CHECKPOINT_ARG="--checkpoint_path $CHECKPOINT_PATH"
fi

HF_TOKEN_ARG=""
if [[ -n "$HF_TOKEN_PATH" ]]; then
    HF_TOKEN_ARG="--hf_token $HF_TOKEN_PATH"
fi

TASKS_FLAG=""
if [[ -n "$TASKS_ARG" ]]; then
    TASKS_FLAG="--tasks $TASKS_ARG"
fi

cd "$OPENVLA_PROBE_DIR"

python "$PY_SCRIPT" \
    --base_vlm_path          "$BASE_VLM_PATH" \
    --pretrained_model_path  "$PRETRAINED_MODEL_PATH" \
    --ref_exp_path           "$REF_EXP_PATH" \
    --image_base_dir         "$IMAGE_BASE_DIR" \
    --output_dir             "$OUTPUT_DIR" \
    --layer_indices          -1 \
    $CHECKPOINT_ARG \
    $HF_TOKEN_ARG \
    $TASKS_FLAG
