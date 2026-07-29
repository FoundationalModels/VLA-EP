#!/usr/bin/env bash
# Capture the language each VLA checkpoint actually generates for the ref_exp
# instructions, as opposed to the hidden states the embedding scripts read.
#
# Runs by default:
#   reasoner   Embodied-CoT steerable-policy-embodied-reasoner-7b (train_reasoner: true)
#   steerable  Embodied-CoT steerable-policy-openvla-7b-bridge    (no train_reasoner)
#   pretrained our bridge-pretrained checkpoint — the control. Trained to emit
#              actions only, so an empty language span here is the expected
#              result and confirms the text/action split is working.
#
# The first two are an unusually tight pair: identical base_vlm, data_mix,
# learning rate, batch size and step count (step-080000-epoch-09), differing
# only in the train_reasoner flag.
#
# Output: visualizations/language/{name}_language.json, plus a console preview.
#
# Usage:
#   bash scripts/get_language_outputs.sh
#   bash scripts/get_language_outputs.sh --max_new_tokens 512

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENVLA_PROBE_DIR="$REPO_DIR/openvla-probe"
CONDA_ENV="openvla-probe"
PY_SCRIPT="$OPENVLA_PROBE_DIR/scripts/get_vla_language_output.py"

CKPT_DIR="$REPO_DIR/checkpoints"
REF_EXP_PATH="$REPO_DIR/media/ref_exp.jsonc"
IMAGE_BASE_DIR="$REPO_DIR/media"
OUTPUT_DIR="$REPO_DIR/visualizations/language"
MODEL_PATH="$CKPT_DIR/hf/openvla-7b"
MAX_NEW_TOKENS=256
# No prime needed. The repo's own prompt elicits reasoning on its own, provided
# the empty token (29871) is NOT appended: that token reproduces action-prediction
# inputs and makes the model skip straight to the action chunk. The reasoner
# opens its chain with "RATIONALE:" unprompted.
PRIME=""
TASKS=(put_carrot_on_plate put_knife_on_plate)

REASONER="$CKPT_DIR/openvla_reasoner_bridge_step-080000-epoch-09-loss=0.0408.pt"
STEERABLE="$CKPT_DIR/openvla_steerable_bridge_step-080000-epoch-09-loss=0.0506.pt"
PRETRAINED="$CKPT_DIR/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt"

usage() { grep '^#' "$0" | sed 's/^# \?//'; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --prime)          PRIME="$2";          shift 2 ;;
        --model_path)     MODEL_PATH="$2";     shift 2 ;;
        --tasks) shift; TASKS=(); while [[ $# -gt 0 && "$1" != --* ]]; do TASKS+=("$1"); shift; done ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

CKPTS=()
[[ -e "$REASONER"   ]] && CKPTS+=("reasoner=$(realpath -s "$REASONER")")
[[ -e "$STEERABLE"  ]] && CKPTS+=("steerable=$(realpath -s "$STEERABLE")")
[[ -e "$PRETRAINED" ]] && CKPTS+=("pretrained=$(realpath -s "$PRETRAINED")")

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "Error: none of the expected checkpoints were found under $CKPT_DIR"
    exit 1
fi

echo "============================================"
echo "VLA language-output capture"
echo "  arch/processor  : $MODEL_PATH"
echo "  checkpoints     : ${CKPTS[*]}"
echo "  tasks           : ${TASKS[*]}"
echo "  max_new_tokens  : $MAX_NEW_TOKENS"
echo "  prime           : ${PRIME@Q}"
echo "  output dir      : $OUTPUT_DIR"
echo "============================================"

source "$SCRIPT_DIR/_activate_conda.sh"
_activate_conda "$CONDA_ENV"

export LD_LIBRARY_PATH="/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$OPENVLA_PROBE_DIR"

python "$PY_SCRIPT" \
    --pretrained_model_path "$MODEL_PATH" \
    --checkpoints           "${CKPTS[@]}" \
    --ref_exp_path          "$REF_EXP_PATH" \
    --image_base_dir        "$IMAGE_BASE_DIR" \
    --output_dir            "$OUTPUT_DIR" \
    --tasks                 "${TASKS[@]}" \
    --max_new_tokens        "$MAX_NEW_TOKENS" \
    --prime                 "$PRIME"

echo ""
echo "Done. Language outputs saved under $OUTPUT_DIR"
