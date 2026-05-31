#!/usr/bin/env bash
# Extract OpenVLA VLM embeddings for all tasks in ref_exp.jsonc.
#
# Always extracts all three weight tiers for every task:
#   1. Base Prismatic VLM   (prism-dinosiglip-224px+7b, pre-robot-training)
#   2. Bridge-pretrained    (openvla_pretrained_bridge checkpoint)
#   3. Co-trained           (task-group-specific fine-tuned checkpoint)
#
# Tasks are split into two groups, each with its own co-trained checkpoint:
#   carrot/knife: put_carrot_on_plate, put_knife_on_plate
#   pot/plate:    flip_pot_upright, put_plate_in_sink
#
# Both mean-pooled and final-token embeddings are extracted in a single forward pass.
#
# Output .npz keys per task:
#   {inst_key}_layer{idx}_mean    — mean-pooled hidden state
#   {inst_key}_layer{idx}_final   — final-token hidden state
#
# Usage:
#   ./get_openvla_vlm_embds.sh
#
#   # override checkpoint paths
#   ./get_openvla_vlm_embds.sh \
#       --carrot_knife_checkpoint ./checkpoints/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt \
#       --pot_plate_checkpoint    ./checkpoints/openvla_cotrained_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt
#
#   # all options
#   ./get_openvla_vlm_embds.sh \
#       --base_vlm_path ./checkpoints/openvla_basevlm_prism-dinosiglip-224px+7b.pt \
#       --pretrained_model_path openvla/openvla-7b \
#       --pretrained_pt_path ./checkpoints/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt \
#       --carrot_knife_checkpoint ./checkpoints/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt \
#       --pot_plate_checkpoint    ./checkpoints/openvla_cotrained_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt \
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
BASE_VLM_PATH="$REPO_DIR/checkpoints/openvla_basevlm_prism-dinosiglip-224px+7b.pt"
PRETRAINED_MODEL_PATH="openvla/openvla-7b"
PRETRAINED_PT_PATH="$REPO_DIR/checkpoints/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt"
CARROT_KNIFE_CHECKPOINT="$REPO_DIR/checkpoints/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt"
POT_PLATE_CHECKPOINT="$REPO_DIR/checkpoints/openvla_cotrained_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt"
HF_TOKEN_PATH=""

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_vlm_path)          BASE_VLM_PATH="$2";             shift 2 ;;
        --pretrained_model_path)  PRETRAINED_MODEL_PATH="$2";     shift 2 ;;
        --pretrained_pt_path)     PRETRAINED_PT_PATH="$2";        shift 2 ;;
        --carrot_knife_checkpoint) CARROT_KNIFE_CHECKPOINT="$2";  shift 2 ;;
        --pot_plate_checkpoint)    POT_PLATE_CHECKPOINT="$2";     shift 2 ;;
        --hf_token)               HF_TOKEN_PATH="$2";             shift 2 ;;
        --output_dir)             OUTPUT_DIR="$2";                shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ! -f "$REF_EXP_PATH" ]]; then
    echo "Error: ref_exp.jsonc not found at $REF_EXP_PATH"
    exit 1
fi

for ckpt in "$CARROT_KNIFE_CHECKPOINT" "$POT_PLATE_CHECKPOINT"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "Error: checkpoint not found: $ckpt"
        exit 1
    fi
done

# Resolve all paths to absolute before cd'ing into OPENVLA_PROBE_DIR
PRETRAINED_PT_PATH="$(realpath "$PRETRAINED_PT_PATH")"
CARROT_KNIFE_CHECKPOINT="$(realpath "$CARROT_KNIFE_CHECKPOINT")"
POT_PLATE_CHECKPOINT="$(realpath "$POT_PLATE_CHECKPOINT")"

echo "============================================"
echo "OpenVLA VLM embedding extraction"
echo "  base VLM                : $BASE_VLM_PATH"
echo "  HF config/arch          : $PRETRAINED_MODEL_PATH"
echo "  pretrained .pt          : $PRETRAINED_PT_PATH"
echo "  carrot/knife co-trained : $CARROT_KNIFE_CHECKPOINT"
echo "  pot/plate co-trained    : $POT_PLATE_CHECKPOINT"
echo "  ref_exp                 : $REF_EXP_PATH"
echo "  images from             : $IMAGE_BASE_DIR"
echo "  output dir              : $OUTPUT_DIR"
echo "  pooling                 : mean + final"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

HF_TOKEN_ARG=""
if [[ -n "$HF_TOKEN_PATH" ]]; then
    HF_TOKEN_ARG="--hf_token $HF_TOKEN_PATH"
fi

cd "$OPENVLA_PROBE_DIR"

COMMON_ARGS=(
    --base_vlm_path          "$BASE_VLM_PATH"
    --pretrained_model_path  "$PRETRAINED_MODEL_PATH"
    --pretrained_pt_path     "$PRETRAINED_PT_PATH"
    --ref_exp_path           "$REF_EXP_PATH"
    --image_base_dir         "$IMAGE_BASE_DIR"
    --output_dir             "$OUTPUT_DIR"
    --layer_indices          -1
)
if [[ -n "$HF_TOKEN_ARG" ]]; then
    COMMON_ARGS+=($HF_TOKEN_ARG)
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
