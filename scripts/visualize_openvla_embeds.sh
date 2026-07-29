#!/usr/bin/env bash
# Visualize OpenVLA VLM embeddings (T-SNE, PCA, cosine-similarity heatmaps).
#
# Runs one comparison per subdirectory:
#   base+pretrained/              — Prismatic base VLM vs Open-X post-trained
#   pretrained+cotrained/         — Open-X post-trained vs fine-tuned checkpoint
#   base+cotrained/               — Prismatic base VLM vs fine-tuned checkpoint
#   pretrained+cotrained_vqa/     — post-trained vs VQA co-trained checkpoint
#   cotrained+cotrained_vqa/      — plain co-trained vs VQA co-trained (the
#                                   comparison that isolates what VQA adds)
#   pretrained+steerable/         — bridge-pretrained vs Embodied-CoT steerable
#                                   policy; both bridge-trained from the same
#                                   base VLM, so architecture is held fixed
#   base+pretrained+cotrained/    — all weight sets together; this is also the
#                                   only run that emits the pairwise-PCA and
#                                   discrimination figures, which need every
#                                   weight set present at once
#
# Each comparison generates both embedding types (layer-1_mean, layer-1_final)
# for all-tasks combined AND for each individual task that has embeddings.
#
# Tasks with no embeddings on disk are skipped automatically, so this works with
# a partial extraction (e.g. carrot/knife only).
#
# Alongside each chart, the numbers behind it are written as JSON:
#   all_tasks_{emb_type}_cosim.json                — full cosine-similarity matrix
#   all_tasks_{emb_type}_task_discrimination.json  — separation / discrimination /
#                                                    sensitivity per weight set
#
# Output layout:
#   visualizations/openvla/<comparison>/
#       all_tasks/  put_carrot_on_plate/  put_knife_on_plate/
#       flip_pot_upright/  put_plate_in_sink/
#
# Usage:
#   ./visualize_openvla_embeds.sh
#   ./visualize_openvla_embeds.sh --embeddings_dir /path/to/embeddings/openvla
#   ./visualize_openvla_embeds.sh --output_dir /path/to/visualizations/openvla

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV="openvla-probe"
PY_SCRIPT="$SCRIPT_DIR/visualize_vlm_embds.py"

EMBEDDINGS_DIR="$REPO_DIR/embeddings/openvla"
OUTPUT_DIR="$REPO_DIR/visualizations/openvla"

ALL_TASKS=(
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

if [[ ! -d "$EMBEDDINGS_DIR" ]]; then
    echo "Error: embeddings dir not found: $EMBEDDINGS_DIR"
    echo "Run scripts/get_openvla_vlm_embds.sh first."
    exit 1
fi

# Only visualize tasks that actually have .npz files, so a partial extraction
# (e.g. carrot/knife only) doesn't produce a run full of "no data" warnings.
TASKS=()
for task in "${ALL_TASKS[@]}"; do
    if compgen -G "$EMBEDDINGS_DIR/$task/${task}_*.npz" > /dev/null; then
        TASKS+=("$task")
    fi
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "Error: no embedding files found under $EMBEDDINGS_DIR"
    exit 1
fi

echo "============================================"
echo "OpenVLA VLM embedding visualization"
echo "  embeddings : $EMBEDDINGS_DIR"
echo "  output     : $OUTPUT_DIR"
echo "  tasks      : ${TASKS[*]}"
echo "============================================"

source "$SCRIPT_DIR/_activate_conda.sh"
_activate_conda "$CONDA_ENV"

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
run_subset base+pretrained \
    --file_suffixes _basevlm_embds _pretrained_embds

run_subset pretrained+cotrained \
    --file_suffixes _pretrained_embds _finetuned_embds

run_subset base+cotrained \
    --file_suffixes _basevlm_embds _finetuned_embds

# ── VQA co-trained comparisons ─────────────────────────────────────────────────
run_subset pretrained+cotrained_vqa \
    --file_suffixes _pretrained_embds _vqafinetuned_embds

run_subset cotrained+cotrained_vqa \
    --file_suffixes _finetuned_embds _vqafinetuned_embds

# ── Steerable policy ───────────────────────────────────────────────────────────
# Both are bridge-trained from the same base VLM, so this pair isolates the
# training recipe with architecture held fixed.
run_subset pretrained+steerable \
    --file_suffixes _pretrained_embds _steerable_embds

# Only train_reasoner differs between these two — everything else (base VLM,
# Bridge V2 data, LR, batch size, step count) is identical.
run_subset steerable+reasoner \
    --file_suffixes _steerable_embds _reasoner_embds

# ── All weight sets together ───────────────────────────────────────────────────
# No --file_suffixes: this also triggers the pairwise-PCA and discrimination
# figures, which are gated on every weight set being present.
run_subset base+pretrained+cotrained

echo ""
echo "Done. Visualizations saved under $OUTPUT_DIR"
