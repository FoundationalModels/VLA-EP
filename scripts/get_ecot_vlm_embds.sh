#!/usr/bin/env bash
# Extract VLM embeddings from Embodied-CoT/ecot-openvla-7b-bridge.
#
# Why this is a separate script from get_openvla_vlm_embds.sh
# -----------------------------------------------------------
# Despite the name, ecot-openvla-7b-bridge is not the same architecture as the
# OpenVLA family in checkpoints/. Its config differs on both backbones:
#
#                       ecot-openvla-7b-bridge      openvla-7b
#   llm_backbone_id     vicuna-v15-7b               llama2-7b-pure
#   vision_backbone_id  siglip-vit-so400m           dinosiglip-vit-so-224px
#   fused vision        False                       True
#   arch_specifier      no-align+gelu-mlp           no-align+fused-gelu-mlp
#   image_resize        letterbox                   resize-naive
#
# It is built on prism-siglip + Vicuna (the OpenVLA v0.1 lineage), so it cannot
# share a model instance with the dinosiglip + Llama-2 checkpoints, and its
# embeddings share no basis with theirs. Compare it through the cosine-based
# metrics (separation / discrimination / sensitivity), which are computed within
# a single model's own space and so travel across architectures — not through a
# joint PCA or a cross-model cosine heatmap, which would be meaningless.
#
# Weights come from the repo's own safetensors, so no .pt tiers are involved:
# the "pretrained" phase is the only one that runs.
#
# Prompt format
# -------------
# The Vicuna-v1.5 backbone is instruction-tuned on the "USER: ... ASSISTANT:"
# template, which get_vla_action() emits for --base_vla_name openvla-v01; the
# plain "openvla" path emits Llama-2-style "In: ... Out:" instead. We extract
# under BOTH so the effect of that choice is measured rather than assumed.
# The v01 run is the primary; the plain run lands in a _plainprompt directory.
#
# Usage:
#   bash scripts/get_ecot_vlm_embds.sh
#   bash scripts/get_ecot_vlm_embds.sh --tasks put_carrot_on_plate put_knife_on_plate

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENVLA_PROBE_DIR="$REPO_DIR/openvla-probe"
CONDA_ENV="openvla-probe"
PY_SCRIPT="$OPENVLA_PROBE_DIR/scripts/get_openvla_vlm_embeds.py"

REF_EXP_PATH="$REPO_DIR/media/ref_exp.jsonc"
IMAGE_BASE_DIR="$REPO_DIR/media"
MODEL_PATH="$REPO_DIR/checkpoints/hf/ecot-openvla-7b-bridge"
OUTPUT_DIR="$REPO_DIR/embeddings/ecot"
TASKS=(put_carrot_on_plate put_knife_on_plate)

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tasks) shift; TASKS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TASKS+=("$1"); shift; done ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: ECoT model directory not found: $MODEL_PATH"
    echo "Download with:"
    echo "  huggingface-cli download Embodied-CoT/ecot-openvla-7b-bridge --local-dir $MODEL_PATH"
    exit 1
fi
MODEL_PATH="$(realpath "$MODEL_PATH")"

echo "============================================"
echo "ECoT VLM embedding extraction"
echo "  model      : $MODEL_PATH"
echo "  tasks      : ${TASKS[*]}"
echo "  output dir : $OUTPUT_DIR"
echo "============================================"

source "$SCRIPT_DIR/_activate_conda.sh"
_activate_conda "$CONDA_ENV"

export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$OPENVLA_PROBE_DIR"

# run_prompt_format BASE_VLA_NAME OUTPUT_DIR
run_prompt_format() {
    local vla_name="$1"; local out="$2"
    echo ""
    echo "========================================"
    echo "Prompt format: $vla_name  ->  $out"
    echo "========================================"
    # No --base_vlm_path and no --pretrained_pt_path: weights load straight from
    # the repo's safetensors, so only the pretrained phase runs.
    python "$PY_SCRIPT" \
        --pretrained_model_path "$MODEL_PATH" \
        --base_vla_name         "$vla_name" \
        --ref_exp_path          "$REF_EXP_PATH" \
        --image_base_dir        "$IMAGE_BASE_DIR" \
        --output_dir            "$out" \
        --layer_indices         -1 \
        --tasks                 "${TASKS[@]}"
}

run_prompt_format openvla-v01 "$OUTPUT_DIR"
run_prompt_format openvla     "${OUTPUT_DIR}_plainprompt"

echo ""
echo "Done."
echo "  primary (Vicuna USER/ASSISTANT format) : $OUTPUT_DIR"
echo "  control (Llama-2 In/Out format)        : ${OUTPUT_DIR}_plainprompt"
