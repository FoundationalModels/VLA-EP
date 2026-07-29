#!/usr/bin/env bash
#SBATCH --job-name=openvla-embds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Extract OpenVLA VLM embeddings for all four weight tiers.
#
# Resource notes:
#   GPU  — the 7B model is held in bf16 (~15 GB). Every GPU on the `gpu`
#          partition (a100 / h100 / h200 / l40s) has enough VRAM, so no GPU type
#          constraint is needed.
#   MEM  — the dominant cost is host-side, not device-side: each co-trained .pt
#          is ~29 GB (fp32) and torch.load pulls the whole thing into CPU RAM
#          before the remapped tensors are copied into the GPU model. 128 GB
#          leaves comfortable headroom.
#   TIME — only 40 forward passes total (4 tiers x 2 tasks x 5 instructions);
#          runtime is dominated by reading ~100 GB of checkpoints off the
#          network filesystem.
#
# Prerequisites (run once, on a login node):
#   bash scripts/setup_openvla_probe_env.sh
#   bash scripts/prefetch_openvla_assets.sh
#
# Submit from the repository root:
#   sbatch slurm/extract_openvla_embds.sbatch

set -eo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"
mkdir -p logs

echo "============================================"
echo "Job              : ${SLURM_JOB_ID:-<interactive>}"
echo "Node             : $(hostname)"
echo "Started          : $(date)"
echo "Repo             : $REPO_DIR"
echo "============================================"

nvidia-smi || echo "(nvidia-smi unavailable)"

module load anaconda/2025.06.0

# Assets were pre-fetched on the login node; make sure a compute node without
# outbound network access fails loudly on a cache miss instead of hanging on a
# connection timeout.
export HF_HOME="/cluster/tufts/hrilab/tdugga02/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# TensorFlow is CPU-only here and only used for image ops; keep it off the GPU
# and quiet in the logs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_CPP_MIN_LOG_LEVEL=2

bash scripts/get_openvla_vlm_embds.sh "$@"

echo ""
echo "Finished: $(date)"
