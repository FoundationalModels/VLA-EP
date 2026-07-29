#!/usr/bin/env bash
#SBATCH --job-name=probe-reason
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
module load anaconda/2025.06.0
export HF_HOME=/cluster/tufts/hrilab/tdugga02/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TF_CPP_MIN_LOG_LEVEL=2
source scripts/_activate_conda.sh; _activate_conda openvla-probe
cd openvla-probe
python scripts/probe_reasoning_prompts.py \
  --pretrained_model_path ../checkpoints/hf/openvla-7b \
  --checkpoint "../checkpoints/openvla_reasoner_bridge_step-080000-epoch-09-loss=0.0408.pt" \
  --ref_exp_path ../media/ref_exp.jsonc \
  --image_base_dir ../media \
  --output ../visualizations/language/reasoning_prompt_probe.json
