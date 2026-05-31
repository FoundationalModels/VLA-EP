# VLA-EP

This repository is a workspace for running embedding extraction and visualization
across three VLA-related projects:

- `open-pi-zero`
- `openvla-mini`
- `openvla-probe`

The projects are checked out as git submodules. Root-level helper scripts in
`scripts/` call into the submodules, read checkpoints from `checkpoints/`, write
embeddings to `embeddings/`, and write visualization outputs to
`visualizations/` when invoked with the commands below.

## 1. Clone And Set Up Submodules

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd VLA-EP
```

If you already cloned the repository without submodules, initialize them from
the repo root:

```bash
git submodule update --init --recursive
```

After pulling updates to this repository, refresh the checked-out submodule
commits with:

```bash
git submodule update --init --recursive
git submodule status
```

## 2. Install Environments

Each submodule has its own dependency setup. Follow the README inside each
project before running the root scripts:

```bash
cd open-pi-zero
# follow open-pi-zero/README.md

cd ../openvla-mini
# follow openvla-mini/README.md

cd ../openvla-probe
# follow openvla-probe/README.md

cd ..
```

The root scripts expect these environments by default:

- `scripts/get_openvla_vlm_embds.sh`: conda env `openvla-probe`
- `scripts/get_minivla_vlm_embds.sh`: conda env `openvla-mini`
- `scripts/get_pi0_vlm_embds.sh`: virtualenv at `open-pi-zero/.venv`
- Visualization scripts: conda envs named in the individual wrapper scripts

## 3. Put Checkpoints In `checkpoints/`

The wrappers have default checkpoint names baked in. Either use those names or
pass explicit checkpoint paths when running the scripts.

Expected OpenVLA defaults:

```text
checkpoints/openvla_basevlm_prism-dinosiglip-224px+7b.pt
checkpoints/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt
checkpoints/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt
checkpoints/openvla_cotrained_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt
```

Expected MiniVLA defaults:

```text
checkpoints/minivla_basevlm_prism-qwen25-extra-dinosiglip-224px+0_5b.pt
checkpoints/minivla_pretrained_bridge_step-362500-epoch-21-loss=0.2259.pt
checkpoints/minivla_cotrained_bridge+sink_carrot+knife_step-010000-epoch-113-loss=0.7581.pt
checkpoints/minivla_cotrained_bridge+sink_pot-plate_step-010000-epoch-53-loss=0.3893.pt
```

Expected Pi0 defaults:

```text
checkpoints/pi0_basevlm_paligemma-3b-pt-224/
checkpoints/pi0_pretrained_bridge_beta_step19296_2024-12-26_22-30_42.pt
checkpoints/pi0_cotrained_bridge+sink_carrot-knife_100000_noaug.pt
checkpoints/pi0_cotrained_bridge+sink_pot-plate_100000_noaug.pt
```

## 4. Extract Embeddings

Run all commands from the repository root. The reference tasks and images are
read from `media/ref_exp.jsonc` and `media/`.

### OpenVLA

```bash
bash scripts/get_openvla_vlm_embds.sh
```

To point the script at different checkpoints:

```bash
bash scripts/get_openvla_vlm_embds.sh \
  --base_vlm_path ./checkpoints/openvla_basevlm_prism-dinosiglip-224px+7b.pt \
  --pretrained_pt_path ./checkpoints/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt \
  --carrot_knife_checkpoint ./checkpoints/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt \
  --pot_plate_checkpoint ./checkpoints/openvla_cotrained_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt \
  --output_dir ./embeddings/openvla
```

If the Hugging Face OpenVLA model requires authentication, also pass:

```bash
--hf_token /path/to/.hf_token
```

### MiniVLA

```bash
bash scripts/get_minivla_vlm_embds.sh
```

To point the script at different checkpoints:

```bash
bash scripts/get_minivla_vlm_embds.sh \
  --base_vlm_path ./checkpoints/minivla_basevlm_prism-qwen25-extra-dinosiglip-224px+0_5b.pt \
  --pretrained_pt_path ./checkpoints/minivla_pretrained_bridge_step-362500-epoch-21-loss=0.2259.pt \
  --carrot_knife_checkpoint ./checkpoints/minivla_cotrained_bridge+sink_carrot+knife_step-010000-epoch-113-loss=0.7581.pt \
  --pot_plate_checkpoint ./checkpoints/minivla_cotrained_bridge+sink_pot-plate_step-010000-epoch-53-loss=0.3893.pt \
  --output_dir ./embeddings/minivla
```

### Pi0

```bash
bash scripts/get_pi0_vlm_embds.sh
```

To point the script at different checkpoints:

```bash
bash scripts/get_pi0_vlm_embds.sh \
  --pretrained_model_path ./checkpoints/pi0_basevlm_paligemma-3b-pt-224 \
  --pi0_pretrained_path   ./checkpoints/pi0_pretrained_bridge_beta_step19296_2024-12-26_22-30_42.pt \
  --carrot_knife_checkpoint ./checkpoints/pi0_cotrained_bridge+sink_carrot-knife_100000_noaug.pt \
  --pot_plate_checkpoint    ./checkpoints/pi0_cotrained_bridge+sink_pot-plate_100000_noaug.pt \
  --output_dir ./embeddings/pi0
```

## 5. Visualize Embeddings

The visualization wrappers read from `embeddings/<model>/` and produce T-SNE,
PCA, and cosine-similarity plots.

```bash
bash scripts/visualize_openvla_embeds.sh \
  --embeddings_dir ./embeddings/openvla \
  --output_dir ./visualization/openvla

bash scripts/visualize_minivla_embeds.sh \
  --embeddings_dir ./embeddings/minivla \
  --output_dir ./visualization/minivla

bash scripts/visualize_pi0_embeds.sh \
  --embeddings_dir ./embeddings/pi0 \
  --output_dir ./visualization/pi0
```

Each model gets comparison subdirectories such as:

```text
visualization/<model>/base+pretrained/
visualization/<model>/pretrained+cotrained/
visualization/<model>/base+cotrained/
visualization/<model>/base+pretrained+cotrained/
```

Within each comparison directory, plots are split into `all_tasks/` and
per-task directories.

## Notes

- Root helper scripts live in `scripts/`.
- Run scripts from the repository root so relative paths resolve correctly.
