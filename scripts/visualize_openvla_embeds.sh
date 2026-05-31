#!/usr/bin/env bash
# Visualize OpenVLA VLM embeddings (T-SNE, PCA, cosine-similarity heatmaps).
#
# Runs four pairwise/combined comparisons, each in its own subdirectory:
#   basevlm+pretrained/          — Prismatic base VLM vs Open-X post-trained
#   pretrained+cotrained/        — Open-X post-trained vs fine-tuned checkpoint
#   basevlm+cotrained/           — Prismatic base VLM vs fine-tuned checkpoint
#   basevlm+pretrained+cotrained/ — all three weight sets together
#
# Each comparison generates both embedding types (layer-1_mean, layer-1_final)
# for all-tasks combined AND for each individual task.
#
# Co-trained checkpoints used per task:
#   put_carrot_on_plate, put_knife_on_plate:
#     openvla_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt
#   flip_pot_upright, put_plate_in_sink:
#     openvla_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt
#
# Output layout:
#   visualization/openvla/
#     {comparison}/
#       all_tasks/
#       put_carrot_on_plate/
#       put_knife_on_plate/
#       flip_pot_upright/
#       put_plate_in_sink/
#
# Usage:
#   ./visualize_openvla_embeds.sh
#   ./visualize_openvla_embeds.sh --embeddings_dir /path/to/embeddings/openvla
#   ./visualize_openvla_embeds.sh --output_dir /path/to/visualization/openvla

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="openvla-probe"
PY_SCRIPT="$SCRIPT_DIR/visualize_vlm_embds.py"

EMBEDDINGS_DIR="$REPO_DIR/embeddings/openvla"
OUTPUT_DIR="$REPO_DIR/visualization/openvla"

TASKS=(
    put_carrot_on_plate
    put_knife_on_plate
    flip_pot_upright
    put_plate_in_sink
)

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --embeddings_dir) EMBEDDINGS_DIR="$2"; shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

echo "============================================"
echo "OpenVLA VLM embedding visualization"
echo "  embeddings : $EMBEDDINGS_DIR"
echo "  output     : $OUTPUT_DIR"
echo "============================================"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# run_subset LABEL [--file_suffixes SUFFIX...]
# Generates all-tasks + per-task plots for the given weight-set subset.
run_subset() {
    local label="$1"; shift
    local subset_dir="$OUTPUT_DIR/$label"

    echo ""
    echo "========================================"
    echo "Subset: $label"
    echo "========================================"

    local common_args=(
        --model openvla
        --embeddings_dir "$EMBEDDINGS_DIR"
        --output_dir "$subset_dir"
    )

    echo "  >>> all tasks combined"
    python "$PY_SCRIPT" "${common_args[@]}" "$@"

    for task in "${TASKS[@]}"; do
        echo "  >>> $task"
        # task_name positional must come before --file_suffixes to avoid nargs="+" greedily
        # consuming it
        python "$PY_SCRIPT" "${common_args[@]}" "$task" "$@"
    done
}

# ── Pairwise comparisons ───────────────────────────────────────────────────────
run_subset basevlm+pretrained \
    --file_suffixes _basevlm_embds _pretrained_embds

run_subset pretrained+cotrained \
    --file_suffixes _pretrained_embds _finetuned_embds

run_subset basevlm+cotrained \
    --file_suffixes _basevlm_embds _finetuned_embds

# ── All three together ─────────────────────────────────────────────────────────
run_subset basevlm+pretrained+cotrained

echo ""
echo "Done. Visualizations saved under $OUTPUT_DIR"
