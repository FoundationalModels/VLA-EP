#!/usr/bin/env bash
#SBATCH --job-name=vla-language
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
module load anaconda/2025.06.0
export HF_HOME=/cluster/tufts/hrilab/tdugga02/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TF_CPP_MIN_LOG_LEVEL=2
bash scripts/get_language_outputs.sh "$@"
