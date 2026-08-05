#!/usr/bin/env bash
# Extract OpenVLA VLM embeddings for the tasks in ref_exp.jsonc.
#
# Extracts up to five weight tiers for every task:
#   1. Base Prismatic VLM   (prism-dinosiglip-224px+7b, pre-robot-training)
#   2. Bridge-pretrained    (openvla_pretrained_bridge checkpoint)
#   3. Co-trained           (task-group-specific fine-tuned checkpoint)
#   4. Co-trained + VQA     (same, but additionally co-trained on LLaVA VQA data)
#   5. Steerable            (Embodied-CoT steerable-policy-openvla-7b-bridge)
#   6. Open-X               (official openvla/openvla-7b release)
#
# Tiers 1, 2, 5 and 6 are shared across all tasks; tiers 3 and 4 are per task group.
#
# All five share the prism-dinosiglip-224px+7b base VLM and load into one model
# instance. Embodied-CoT's *ecot*-openvla-7b-bridge does NOT belong here — it is
# built on prism-siglip + Vicuna-7B, so it needs its own model instantiation and
# is extracted separately into embeddings/ecot/ (see scripts/get_ecot_vlm_embds.sh).
#
# Tasks are split into two groups, each with its own co-trained checkpoint(s):
#   carrot/knife: put_carrot_on_plate, put_knife_on_plate
#   pot/plate:    flip_pot_upright, put_plate_in_sink
#
# A group is skipped when its co-trained checkpoint is not present, so this
# script works with a partial set of checkpoints (currently only carrot/knife
# has been downloaded). The VQA checkpoint for a group is optional and simply
# omitted when absent.
#
# Both mean-pooled and final-token embeddings are extracted in a single forward pass.
#
# Output .npz keys per task:
#   {inst_key}_layer{idx}_mean    — mean-pooled hidden state
#   {inst_key}_layer{idx}_final   — final-token hidden state
#
# Output files per task, under {output_dir}/{task}/:
#   {task}_{stem}_basevlm_embds.npz
#   {task}_{stem}_pretrained_embds.npz
#   {task}_{stem}_finetuned_embds.npz
#   {task}_{stem}_vqafinetuned_embds.npz
#
# Usage:
#   ./get_openvla_vlm_embds.sh
#
#   # override checkpoint paths
#   ./get_openvla_vlm_embds.sh \
#       --carrot_knife_checkpoint     ./checkpoints/openvla_cotrained_....pt \
#       --carrot_knife_vqa_checkpoint ./checkpoints/openvla_vqa_....pt
#
#   # all options
#   ./get_openvla_vlm_embds.sh \
#       --base_vlm_path ./checkpoints/openvla_basevlm_prism-dinosiglip-224px+7b.pt \
#       --pretrained_model_path openvla/openvla-7b \
#       --pretrained_pt_path ./checkpoints/openvla_pretrained_bridge_....pt \
#       --carrot_knife_checkpoint     ./checkpoints/openvla_cotrained_carrot-knife_....pt \
#       --carrot_knife_vqa_checkpoint ./checkpoints/openvla_vqa_....pt \
#       --pot_plate_checkpoint        ./checkpoints/openvla_cotrained_pot-plate_....pt \
#       --pot_plate_vqa_checkpoint    ./checkpoints/openvla_vqa_pot-plate_....pt \
#       --hf_token /path/to/.hf_token

# No `set -u`: the cluster anaconda module's deactivate hook dereferences an
# unset variable, which would abort the script at `conda activate`.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENVLA_PROBE_DIR="$REPO_DIR/openvla-probe"
CONDA_ENV="openvla-probe"
PY_SCRIPT="$OPENVLA_PROBE_DIR/scripts/get_openvla_vlm_embeds.py"

REF_EXP_PATH="$REPO_DIR/media/ref_exp.jsonc"
IMAGE_BASE_DIR="$REPO_DIR/media"
OUTPUT_DIR="$REPO_DIR/embeddings/openvla"
CKPT_DIR="$REPO_DIR/checkpoints"

BASE_VLM_PATH="$CKPT_DIR/openvla_basevlm_prism-dinosiglip-224px+7b.pt"
# Local snapshot of openvla/openvla-7b holding config + processor only (no
# safetensors) — see scripts/prefetch_openvla_assets.sh. The extraction script
# builds the architecture from config and loads every weight tier from .pt files,
# so the 15 GB of HF weights are never needed. Falls back to the hub ID if absent.
PRETRAINED_MODEL_PATH="$CKPT_DIR/hf/openvla-7b"
[[ -d "$PRETRAINED_MODEL_PATH" ]] || PRETRAINED_MODEL_PATH="openvla/openvla-7b"

PRETRAINED_PT_PATH="$CKPT_DIR/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt"
CARROT_KNIFE_CHECKPOINT="$CKPT_DIR/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt"
CARROT_KNIFE_VQA_CHECKPOINT="$CKPT_DIR/openvla_vqa_step-005000-epoch-22-loss=0.4131.pt"
POT_PLATE_CHECKPOINT="$CKPT_DIR/openvla_cotrained_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt"
POT_PLATE_VQA_CHECKPOINT=""
# Embodied-CoT steerable policy. Same architecture and base VLM
# (prism-dinosiglip-224px+7b) as everything above and bridge-trained, so it loads
# into the same model instance as an extra weight set. Not task-group specific.
STEERABLE_CHECKPOINT="$CKPT_DIR/openvla_steerable_bridge_step-080000-epoch-09-loss=0.0506.pt"
# Official openvla/openvla-7b release (Open-X trained). Same architecture and base
# VLM as the rest, so it joins the shared space as its own tier. Distinct from the
# "pre-trained" tier above, which is a *bridge*-trained checkpoint from openvla-mini.
OPENX_MODEL_DIR="$CKPT_DIR/hf/openvla-7b-full"
# Embodied-CoT embodied-reasoner: same base VLM, Bridge V2 data, LR, batch and
# step as the steerable checkpoint above, differing only in train_reasoner: true.
REASONER_CHECKPOINT="$CKPT_DIR/openvla_reasoner_bridge_step-080000-epoch-09-loss=0.0408.pt"
# Standard OpenVLA (Open-X release) fine-tuned in-domain on carrot/knife. Same
# Prismatic .pt layout as the rest, stored bf16 rather than fp32.
OXEFT_CHECKPOINT="$CKPT_DIR/openvla_oxe+carrot-knife.pt"
# bridge_llava_7b -- the Bridge-stage checkpoint the VQA co-trained model was
# fine-tuned from. Its presence de-confounds the VQA comparison, which until now
# had to borrow the plain Bridge checkpoint as its baseline.
VQAPRETRAIN_CHECKPOINT="$CKPT_DIR/openvla_vqa_pretrain_step-200000-epoch-09-loss%3D0.2419-bf16.pt"
HF_TOKEN_PATH=""

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_vlm_path)               BASE_VLM_PATH="$2";               shift 2 ;;
        --pretrained_model_path)       PRETRAINED_MODEL_PATH="$2";       shift 2 ;;
        --pretrained_pt_path)          PRETRAINED_PT_PATH="$2";          shift 2 ;;
        --carrot_knife_checkpoint)     CARROT_KNIFE_CHECKPOINT="$2";     shift 2 ;;
        --carrot_knife_vqa_checkpoint) CARROT_KNIFE_VQA_CHECKPOINT="$2"; shift 2 ;;
        --pot_plate_checkpoint)        POT_PLATE_CHECKPOINT="$2";        shift 2 ;;
        --pot_plate_vqa_checkpoint)    POT_PLATE_VQA_CHECKPOINT="$2";    shift 2 ;;
        --steerable_checkpoint)        STEERABLE_CHECKPOINT="$2";        shift 2 ;;
        --openx_model_dir)             OPENX_MODEL_DIR="$2";             shift 2 ;;
        --reasoner_checkpoint)         REASONER_CHECKPOINT="$2";         shift 2 ;;
        --oxeft_checkpoint)            OXEFT_CHECKPOINT="$2";            shift 2 ;;
        --vqa_pretrain_checkpoint)     VQAPRETRAIN_CHECKPOINT="$2";      shift 2 ;;
        --hf_token)                    HF_TOKEN_PATH="$2";               shift 2 ;;
        --output_dir)                  OUTPUT_DIR="$2";                  shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

if [[ ! -f "$REF_EXP_PATH" ]]; then
    echo "Error: ref_exp.jsonc not found at $REF_EXP_PATH"
    exit 1
fi

for required in "$BASE_VLM_PATH" "$PRETRAINED_PT_PATH"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: checkpoint not found: $required"
        exit 1
    fi
done

if [[ ! -f "$CARROT_KNIFE_CHECKPOINT" && ! -f "$POT_PLATE_CHECKPOINT" ]]; then
    echo "Error: no co-trained checkpoint found for either task group."
    echo "  carrot/knife: $CARROT_KNIFE_CHECKPOINT"
    echo "  pot/plate   : $POT_PLATE_CHECKPOINT"
    exit 1
fi

# Resolve to absolute paths before cd'ing into OPENVLA_PROBE_DIR.
# -s keeps symlinks intact: checkpoints/ may link into the HuggingFace blob
# cache, whose targets are content-hashed filenames. Following them would put an
# opaque hash into every output .npz name and into the plot legends.
abspath() { [[ -n "$1" && -e "$1" ]] && realpath -s "$1" || echo ""; }
BASE_VLM_PATH="$(abspath "$BASE_VLM_PATH")"
PRETRAINED_PT_PATH="$(abspath "$PRETRAINED_PT_PATH")"
CARROT_KNIFE_CHECKPOINT="$(abspath "$CARROT_KNIFE_CHECKPOINT")"
CARROT_KNIFE_VQA_CHECKPOINT="$(abspath "$CARROT_KNIFE_VQA_CHECKPOINT")"
POT_PLATE_CHECKPOINT="$(abspath "$POT_PLATE_CHECKPOINT")"
POT_PLATE_VQA_CHECKPOINT="$(abspath "$POT_PLATE_VQA_CHECKPOINT")"
STEERABLE_CHECKPOINT="$(abspath "$STEERABLE_CHECKPOINT")"
OPENX_MODEL_DIR="$(abspath "$OPENX_MODEL_DIR")"
REASONER_CHECKPOINT="$(abspath "$REASONER_CHECKPOINT")"
OXEFT_CHECKPOINT="$(abspath "$OXEFT_CHECKPOINT")"
VQAPRETRAIN_CHECKPOINT="$(abspath "$VQAPRETRAIN_CHECKPOINT")"
[[ -d "$PRETRAINED_MODEL_PATH" ]] && PRETRAINED_MODEL_PATH="$(realpath "$PRETRAINED_MODEL_PATH")"

echo "============================================"
echo "OpenVLA VLM embedding extraction"
echo "  base VLM                    : $BASE_VLM_PATH"
echo "  HF config/arch              : $PRETRAINED_MODEL_PATH"
echo "  pretrained .pt              : $PRETRAINED_PT_PATH"
echo "  carrot/knife co-trained     : ${CARROT_KNIFE_CHECKPOINT:-(skipped)}"
echo "  carrot/knife co-trained+VQA : ${CARROT_KNIFE_VQA_CHECKPOINT:-(skipped)}"
echo "  pot/plate    co-trained     : ${POT_PLATE_CHECKPOINT:-(skipped)}"
echo "  pot/plate    co-trained+VQA : ${POT_PLATE_VQA_CHECKPOINT:-(skipped)}"
echo "  steerable (all tasks)       : ${STEERABLE_CHECKPOINT:-(skipped)}"
echo "  Open-X openvla-7b (all)     : ${OPENX_MODEL_DIR:-(skipped)}"
echo "  reasoner (all tasks)        : ${REASONER_CHECKPOINT:-(skipped)}"
echo "  Open-X + carrot/knife FT    : ${OXEFT_CHECKPOINT:-(skipped)}"
echo "  Bridge+VQA pretrained       : ${VQAPRETRAIN_CHECKPOINT:-(skipped)}"
echo "  ref_exp                     : $REF_EXP_PATH"
echo "  images from                 : $IMAGE_BASE_DIR"
echo "  output dir                  : $OUTPUT_DIR"
echo "  pooling                     : mean + final"
echo "============================================"

source "$SCRIPT_DIR/_activate_conda.sh"
_activate_conda "$CONDA_ENV"

export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
if [[ -n "$HF_TOKEN_PATH" ]]; then
    COMMON_ARGS+=(--hf_token "$HF_TOKEN_PATH")
fi

# run_group LABEL COTRAINED_CKPT VQA_CKPT TASK...
run_group() {
    local label="$1"; local ckpt="$2"; local vqa_ckpt="$3"; shift 3

    if [[ -z "$ckpt" ]]; then
        echo ""
        echo "Skipping $label — no co-trained checkpoint present."
        return 0
    fi

    echo ""
    echo "========================================"
    echo "Group: $label  ($*)"
    echo "========================================"

    local group_args=("${COMMON_ARGS[@]}" --tasks "$@" --checkpoint_path "$ckpt")
    local extras=()
    [[ -n "$STEERABLE_CHECKPOINT" ]] && extras+=("steerable=$STEERABLE_CHECKPOINT")
    [[ -n "$OPENX_MODEL_DIR"      ]] && extras+=("openx=$OPENX_MODEL_DIR")
    [[ -n "$REASONER_CHECKPOINT"  ]] && extras+=("reasoner=$REASONER_CHECKPOINT")
    [[ -n "$OXEFT_CHECKPOINT"     ]] && extras+=("oxeft=$OXEFT_CHECKPOINT")
    [[ -n "$VQAPRETRAIN_CHECKPOINT" ]] && extras+=("vqapretrain=$VQAPRETRAIN_CHECKPOINT")
    if [[ ${#extras[@]} -gt 0 ]]; then
        group_args+=(--extra_checkpoints "${extras[@]}")
    fi
    if [[ -n "$vqa_ckpt" ]]; then
        group_args+=(--vqa_checkpoint_path "$vqa_ckpt")
    else
        echo "  (no VQA co-trained checkpoint for this group — tier 4 skipped)"
    fi

    python "$PY_SCRIPT" "${group_args[@]}"
}

run_group "carrot/knife" \
    "$CARROT_KNIFE_CHECKPOINT" "$CARROT_KNIFE_VQA_CHECKPOINT" \
    put_carrot_on_plate put_knife_on_plate

run_group "pot/plate" \
    "$POT_PLATE_CHECKPOINT" "$POT_PLATE_VQA_CHECKPOINT" \
    flip_pot_upright put_plate_in_sink

echo ""
echo "Done. Embeddings saved under $OUTPUT_DIR"
