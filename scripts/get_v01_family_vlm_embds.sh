#!/usr/bin/env bash
# Extract embeddings for the OpenVLA-v0.1 family into one shared space.
#
#   openvla/openvla-v01-7b               -> _pretrained_embds  ("OpenVLA-v0.1")
#   Embodied-CoT/ecot-openvla-7b-bridge  -> _ecot_embds        ("ECoT")
#
# Why these two and not ECoT against the main OpenVLA family
# ----------------------------------------------------------
# ecot-openvla-7b-bridge and openvla-v01-7b are an exact architectural match on
# every config field that matters:
#
#   llm_backbone_id           vicuna-v15-7b
#   vision_backbone_id        siglip-vit-so400m
#   arch_specifier            no-align+gelu-mlp
#   image_resize_strategy     letterbox
#   use_fused_vision_backbone False
#   image_sizes               [224]
#
# So both load into a single model instance and their embeddings live in the
# same space — a joint PCA, cosine heatmap, and pairwise projection are all
# valid here. Against the prism-dinosiglip + Llama-2 models in
# get_openvla_vlm_embds.sh they would not be: different LLM weights and a
# different visual encoder mean no shared basis, so a joint projection would
# look interpretable while being an artifact.
#
# Both use the Vicuna "USER: ... ASSISTANT:" prompt template (--base_vla_name
# openvla-v01), which is what the shared backbone was instruction-tuned on.
# Extracting ECoT under the wrong template shifts its metrics by ~3x, so this is
# not a detail — see embeddings/ecot_plainprompt/ for the control.
#
# Usage:
#   bash scripts/get_v01_family_vlm_embds.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENVLA_PROBE_DIR="$REPO_DIR/openvla-probe"
CONDA_ENV="openvla-probe"
PY_SCRIPT="$OPENVLA_PROBE_DIR/scripts/get_openvla_vlm_embeds.py"

REF_EXP_PATH="$REPO_DIR/media/ref_exp.jsonc"
IMAGE_BASE_DIR="$REPO_DIR/media"
V01_PATH="$REPO_DIR/checkpoints/hf/openvla-v01-7b"
ECOT_PATH="$REPO_DIR/checkpoints/hf/ecot-openvla-7b-bridge"
OUTPUT_DIR="$REPO_DIR/embeddings/v01_family"
TASKS=(put_carrot_on_plate put_knife_on_plate)

usage() { grep '^#' "$0" | sed 's/^# \?//'; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --v01_path)   V01_PATH="$2";   shift 2 ;;
        --ecot_path)  ECOT_PATH="$2";  shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tasks) shift; TASKS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TASKS+=("$1"); shift; done ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for d in "$V01_PATH" "$ECOT_PATH"; do
    if [[ ! -d "$d" ]]; then
        echo "Error: model directory not found: $d"
        exit 1
    fi
done
V01_PATH="$(realpath "$V01_PATH")"
ECOT_PATH="$(realpath "$ECOT_PATH")"

echo "============================================"
echo "OpenVLA-v0.1 family embedding extraction"
echo "  baseline (v0.1) : $V01_PATH"
echo "  ECoT            : $ECOT_PATH"
echo "  tasks           : ${TASKS[*]}"
echo "  output dir      : $OUTPUT_DIR"
echo "  prompt template : openvla-v01 (Vicuna USER/ASSISTANT)"
echo "============================================"

source "$SCRIPT_DIR/_activate_conda.sh"
_activate_conda "$CONDA_ENV"

export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$OPENVLA_PROBE_DIR"

# The model instance is built from the v0.1 config and its own safetensors supply
# the baseline tier; ECoT is then loaded into that same instance as an extra set.
python "$PY_SCRIPT" \
    --pretrained_model_path "$V01_PATH" \
    --base_vla_name         openvla-v01 \
    --ref_exp_path          "$REF_EXP_PATH" \
    --image_base_dir        "$IMAGE_BASE_DIR" \
    --output_dir            "$OUTPUT_DIR" \
    --layer_indices         -1 \
    --tasks                 "${TASKS[@]}" \
    --extra_checkpoints     "ecot=$ECOT_PATH"

echo ""
echo "Done. Embeddings saved under $OUTPUT_DIR"
