#!/usr/bin/env bash
# Visualize MiniVLA VLM embeddings (T-SNE, PCA, cosine-similarity heatmaps).
#
# Runs four pairwise/combined comparisons, each in its own subdirectory:
#   base+pretrained/          — Prismatic base VLM vs Bridge-pretrained
#   pretrained+cotrained/     — Bridge-pretrained vs fine-tuned checkpoint
#   base+cotrained/           — Prismatic base VLM vs fine-tuned checkpoint
#   base+pretrained+cotrained/ — all three weight sets together
#
# Each comparison generates both embedding types (layer-1_mean, layer-1_final)
# for all-tasks combined AND for each individual task.
#
# Co-trained checkpoints used per task:
#   put_carrot_on_plate, put_knife_on_plate:
#     checkpoints/minivla_cotrained_bridge+sink_carrot+knife_step-010000-epoch-113-loss=0.7581.pt
#   flip_pot_upright, put_plate_in_sink:
#     checkpoints/minivla_cotrained_bridge+sink_pot-plate_step-010000-epoch-53-loss=0.3893.pt
#
# Output layout:
#   visualizations/minivla/
#     base+pretrained/
#     pretrained+cotrained/
#     base+cotrained/
#     base+pretrained+cotrained/
#       each contains: all_tasks/  put_carrot_on_plate/  put_knife_on_plate/
#                      flip_pot_upright/  put_plate_in_sink/
#
# Usage:
#   ./visualize_minivla_embeds.sh
#   ./visualize_minivla_embeds.sh --embeddings_dir /path/to/embeddings/minivla
#   ./visualize_minivla_embeds.sh --output_dir /path/to/visualizations/minivla

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="openvla-mini"
PY_SCRIPT="$SCRIPT_DIR/visualize_vlm_embds.py"

EMBEDDINGS_DIR="$REPO_DIR/embeddings/minivla"
OUTPUT_DIR="$REPO_DIR/visualizations/minivla"

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
echo "MiniVLA VLM embedding visualization"
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
        --model minivla
        --embeddings_dir "$EMBEDDINGS_DIR"
        --output_dir "$subset_dir"
    )

    echo "  >>> all tasks combined"
    python "$PY_SCRIPT" "${common_args[@]}" "$@"

    for task in "${TASKS[@]}"; do
        echo "  >>> $task"
        python "$PY_SCRIPT" "${common_args[@]}" "$task" "$@"
    done
}

# ── Pairwise comparisons ───────────────────────────────────────────────────────
run_subset base+pretrained \
    --file_suffixes _basevlm_embds _pretrained_embds

run_subset pretrained+cotrained \
    --file_suffixes _pretrained_embds _finetuned_embds

run_subset base+cotrained \
    --file_suffixes _basevlm_embds _finetuned_embds

# ── All three together ─────────────────────────────────────────────────────────
run_subset base+pretrained+cotrained

echo ""
echo "Done. Visualizations saved under $OUTPUT_DIR"
