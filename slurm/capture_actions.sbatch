#!/usr/bin/env bash
#SBATCH --job-name=vla-actions
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
source scripts/_activate_conda.sh; _activate_conda openvla-probe
C=checkpoints
cd openvla-probe
# --pad_empty_token + --max_new_tokens 7 reproduces predict_action() exactly:
# the empty token is appended and only the 7-DoF action chunk is generated, with
# no reasoning preamble. This is the action the policy would actually execute.
python scripts/get_vla_language_output.py \
  --pretrained_model_path ../$C/hf/openvla-7b \
  --checkpoints \
     "base=../$C/openvla_basevlm_prism-dinosiglip-224px+7b.pt" \
     "openx=../$C/hf/openvla-7b-full" \
     "bridge-pretrained=../$C/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt" \
     "ECoT-steerable=../$C/openvla_steerable_bridge_step-080000-epoch-09-loss=0.0506.pt" \
     "ECoT-reasoner=../$C/openvla_reasoner_bridge_step-080000-epoch-09-loss=0.0408.pt" \
     "co-trained=../$C/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt" \
     "co-trained+VQA=../$C/openvla_vqa_step-005000-epoch-22-loss=0.4131.pt" \
  --ref_exp_path ../media/ref_exp.jsonc --image_base_dir ../media \
  --output_dir ../visualizations/actions \
  --tasks put_carrot_on_plate put_knife_on_plate \
  --pad_empty_token --max_new_tokens 7
