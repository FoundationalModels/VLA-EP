#!/usr/bin/env bash
#SBATCH --job-name=openvla-viz
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Render the OpenVLA embedding figures (T-SNE, PCA, cosine-similarity) and write
# the numeric values behind them as JSON.
#
# No GPU: this only reads .npz files and runs sklearn/matplotlib on CPU.
#
# Submit from the repository root, after extraction has finished:
#   sbatch slurm/visualize_openvla_embds.sbatch
#
# Or chain it directly onto the extraction job:
#   jid=$(sbatch --parsable slurm/extract_openvla_embds.sbatch)
#   sbatch --dependency=afterok:$jid slurm/visualize_openvla_embds.sbatch

set -eo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"
mkdir -p logs

echo "============================================"
echo "Job     : ${SLURM_JOB_ID:-<interactive>}"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo "============================================"

module load anaconda/2025.06.0

export MPLCONFIGDIR="/cluster/tufts/hrilab/tdugga02/.cache/matplotlib"
export TF_CPP_MIN_LOG_LEVEL=2

bash scripts/visualize_openvla_embeds.sh "$@"

echo ""
echo "Finished: $(date)"
