#!/usr/bin/env bash
# Visualize Pi0 VLM embeddings (T-SNE, PCA, cosine-similarity heatmaps).
#
# Runs four pairwise/combined comparisons, each in its own subdirectory:
#   base+pretrained/          — PaliGemma base vs Pi0 bridge-pretrained
#   pretrained+cotrained/     — Pi0 bridge-pretrained vs Pi0 task co-trained
#   base+cotrained/           — PaliGemma base vs Pi0 task co-trained
#   base+pretrained+cotrained/ — all three weight sets together
#
# Each comparison generates both embedding types (all_layers, last_layer)
# for all-tasks combined AND for each individual task.
#
# Bridge-pretrained checkpoint:
#   checkpoints/pi0_pretrained_bridge_beta_step19296_2024-12-26_22-30_42.pt
#
# Co-trained checkpoints used per task:
#   put_carrot_on_plate, put_knife_on_plate:
#     checkpoints/pi0_cotrained_bridge+sink_carrot-knife_100000_noaug.pt
#   flip_pot_upright, put_plate_in_sink:
#     checkpoints/pi0_cotrained_bridge+sink_pot-plate_100000_noaug.pt
#
# Output layout:
#   visualizations/pi0/
#     {comparison}/
#       all_tasks/
#       put_carrot_on_plate/
#       put_knife_on_plate/
#       flip_pot_upright/
#       put_plate_in_sink/
#
# Usage:
#   ./visualize_pi0_embeds.sh
#   ./visualize_pi0_embeds.sh --embeddings_dir /path/to/embeddings/pi0
#   ./visualize_pi0_embeds.sh --output_dir /path/to/visualizations/pi0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="openvla-probe"
PY_SCRIPT="$SCRIPT_DIR/visualize_vlm_embds.py"

EMBEDDINGS_DIR="$REPO_DIR/embeddings/pi0"
OUTPUT_DIR="$REPO_DIR/visualizations/pi0"

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
echo "Pi0 VLM embedding visualization"
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
        --model pi0
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
run_subset base+pretrained \
    --file_suffixes _pretrained_embds _pi0pretrained_embds

run_subset pretrained+cotrained \
    --file_suffixes _pi0pretrained_embds _finetuned_embds

run_subset base+cotrained \
    --file_suffixes _pretrained_embds _finetuned_embds

# ── All three together ─────────────────────────────────────────────────────────
run_subset base+pretrained+cotrained

echo ""
echo "Done. Visualizations saved under $OUTPUT_DIR"
