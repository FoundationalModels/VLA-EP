#!/usr/bin/env bash
# Pre-download everything the extraction job fetches from the network, so the
# SLURM job itself can run without internet access.
#
# Two things get cached:
#
#   1. openvla/openvla-7b — config + tokenizer + processor files ONLY.
#      get_vla() builds the architecture from config and every weight tier is
#      then loaded from a local .pt, so the repo's ~15 GB of safetensors are
#      never read. Omitting them makes from_pretrained raise OSError, which
#      get_vla() already handles by falling back to config-only instantiation
#      (the "safetensors deleted to save disk space" path).
#
#   2. DINOv2 + SigLIP TIMM weights. The base Prismatic .pt stores only the
#      projector and LLM backbone — the vision backbone is frozen during
#      Prismatic pre-training and never saved — so load_pretrained_vision_backbone()
#      reconstructs it via timm with pretrained=True on every base-VLM run.
#
# Usage (login node, needs internet):
#   bash scripts/prefetch_openvla_assets.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROBE_DIR="$REPO_DIR/openvla-probe"
DEST="$REPO_DIR/checkpoints/hf/openvla-7b"

export HF_HOME="${HF_HOME:-/cluster/tufts/hrilab/tdugga02/.cache/huggingface}"

module load anaconda/2025.06.0
eval "$(conda shell.bash hook)"
conda activate openvla-probe

mkdir -p "$DEST"

echo "[*] HF_HOME = $HF_HOME"
echo "[*] Fetching openvla/openvla-7b config + processor files → $DEST"

python - <<PY
import os
from huggingface_hub import snapshot_download

dest = snapshot_download(
    "openvla/openvla-7b",
    local_dir="$DEST",
    # Everything except the model weights. The safetensors *index* has to go
    # too: left behind, from_pretrained would find it, then fail looking for
    # shards that were never downloaded, instead of taking the clean
    # "no weights present" path that get_vla() handles.
    ignore_patterns=["*.safetensors", "*.safetensors.index.json",
                     "*.bin", "*.bin.index.json", "*.pt", "*.h5", "*.msgpack"],
)
print(f"  -> {dest}")
for name in sorted(os.listdir(dest)):
    if not name.startswith("."):
        print(f"     {name}")
PY

echo ""
echo "[*] Warming TIMM cache for DINOv2 + SigLIP vision backbone"

cd "$PROBE_DIR"
python - <<'PY'
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone

# pretrained=True inside the backbone triggers the TIMM/HF download we want cached.
vb = DinoSigLIPViTBackbone("dinosiglip-vit-so-224px", "resize-naive")
n_dino = sum(p.numel() for p in vb.dino_featurizer.parameters())
n_siglip = sum(p.numel() for p in vb.siglip_featurizer.parameters())
print(f"  DINOv2  params: {n_dino:,}")
print(f"  SigLIP  params: {n_siglip:,}")
print("  vision backbone weights cached")
PY

echo ""
echo "Done. The extraction job can now run with HF_HUB_OFFLINE=1."
