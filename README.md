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

### OpenVLA on the Tufts HPC

`scripts/setup_openvla_probe_env.sh` builds the `openvla-probe` conda
environment end to end — it encodes the fixes needed on this cluster, so prefer
it over following `openvla-probe/README.md` by hand:

```bash
bash scripts/setup_openvla_probe_env.sh
```

What it handles that the upstream instructions do not:

- The env lives in `/cluster/tufts/hrilab/tdugga02/envs` and is registered via
  `~/.condarc`, because `$HOME` has a 28 GB quota that a torch + TF env exceeds.
- Packages come from `conda-forge` with `--override-channels`; the cluster's
  default `repo.anaconda.com` channels are gated behind a Terms of Service
  prompt that fails non-interactively.
- `flash-attn` is skipped. It needs a long `nvcc` build and `get_vla()` already
  falls back to `sdpa` attention.
- TensorFlow is installed **CPU-only**. `openvla_utils` imports it
  unconditionally but only uses `tf.image`, so the GPU build would just contend
  with torch for VRAM.
- `tensorflow-metadata` is pinned to 1.15.0 — newer releases ship protobuf-5.x
  generated modules that cannot import against the protobuf 4.x TF 2.15 needs.
- `dlimp` and `tensorflow_graphics` are installed `--no-deps`. Embedding
  extraction never builds a dataset, but `prismatic/__init__.py` eagerly imports
  the training stack, so the imports must still resolve.

Then cache the assets that would otherwise be fetched at run time, so the
compute node needs no outbound network:

```bash
bash scripts/prefetch_openvla_assets.sh
```

This downloads `openvla/openvla-7b` **config and processor files only** (~2 MB
rather than ~15 GB — every weight tier is loaded from a local `.pt`, so the HF
safetensors are never read) and warms the TIMM cache for the DINOv2 + SigLIP
vision backbone. That backbone download is expected: the base Prismatic `.pt`
stores only the projector and LLM backbone, since the vision tower is frozen
during Prismatic pre-training and never saved.

## 3. Put Checkpoints In `checkpoints/`

The wrappers have default checkpoint names baked in. Either use those names or
pass explicit checkpoint paths when running the scripts.

Expected OpenVLA defaults:

```text
checkpoints/openvla_basevlm_prism-dinosiglip-224px+7b.pt
checkpoints/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt
checkpoints/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt
checkpoints/openvla_cotrained_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt
checkpoints/openvla_vqa_step-005000-epoch-22-loss=0.4131.pt
```

`openvla_vqa_*` is the LLaVA-VQA co-trained checkpoint (`vl_dataset_id:
llava-v15`, `vl_ratio: 0.2`), giving OpenVLA a fourth weight set alongside the
usual base / pre-trained / co-trained tiers. It is a carrot/knife checkpoint;
there is no pot/plate equivalent yet.

A task group is skipped automatically when its co-trained checkpoint is absent,
so a partial checkpoint set works without editing anything.

### Embodied-CoT models

Two external models from the `Embodied-CoT` org are also wired in, but they are
**not** interchangeable, despite near-identical names:

| | `steerable-policy-openvla-7b-bridge` | `ecot-openvla-7b-bridge` |
|---|---|---|
| base VLM | `prism-dinosiglip-224px+7b` | `prism-siglip` (v0.1 lineage) |
| LLM backbone | Llama-2-7B | **Vicuna-7B-v1.5** |
| vision backbone | DINOv2 + SigLIP fused | **SigLIP only** |
| image preprocessing | resize-naive | **letterbox** |
| format | Prismatic `.pt` | HF safetensors |
| prompt template | `In:/Out:` | **`USER:/ASSISTANT:`** |

`steerable` shares the architecture and base VLM of everything else, so it loads
into the same model instance as a fifth weight set and appears on the shared PCA
and cosine plots. Being bridge-trained, its natural reference is the
bridge-pretrained tier.

`ecot` does not. It is a different lineage on both backbones, so its embeddings
share no basis with the OpenVLA family — a joint PCA or cross-model cosine
heatmap would look interpretable and mean nothing. It gets its own model
instantiation, its own `embeddings/ecot/`, and is compared only through the
cosine-based metrics, which are computed within a single model's space and so
travel across architectures:

```bash
bash scripts/get_ecot_vlm_embds.sh
```

```bash
python scripts/compare_model_metrics.py
```

The prompt template is not a detail here. ECoT is extracted under both
templates, and the same instruction comes out **near-orthogonal** between them
(mean cosine 0.156), shifting the metrics by roughly 3×. `embeddings/ecot/` uses
the Vicuna template that its backbone was instruction-tuned on and is the one to
analyse; `embeddings/ecot_plainprompt/` is kept only as the control that
establishes how much the choice matters.

One caveat worth recording: this VQA checkpoint was trained from
`bridge_llava_7b/step-200000-epoch-09-loss=0.2419.pt`, whereas the plain
co-trained checkpoint came from `bridge_7b/step-200000-epoch-23-loss=0.0370.pt`.
Only the latter is on the cluster, so both co-trained tiers are currently
plotted against that one shared pre-trained baseline. It is a common reference
point, not the VQA checkpoint's literal parent.

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

On the HPC, submit rather than running on a login node — see
[Running on SLURM](#7-running-on-slurm) below. Directly:

```bash
bash scripts/get_openvla_vlm_embds.sh
```

This extracts up to four weight tiers per task in a single pass over one model
instance, each phase overwriting the previous weights in place:

| Tier | Weights | Output suffix |
|---|---|---|
| 1 | base Prismatic VLM | `_basevlm_embds` |
| 2 | Bridge-pretrained | `_pretrained_embds` |
| 3 | co-trained | `_finetuned_embds` |
| 4 | co-trained + LLaVA VQA | `_vqafinetuned_embds` |
| 5 | Embodied-CoT steerable policy | `_steerable_embds` |

Tiers 1, 2 and 5 are shared across all tasks; 3 and 4 are per task group. Any
further same-architecture checkpoint can be added without new flags:

```bash
bash scripts/get_openvla_vlm_embds.sh --extra_checkpoints myname=./checkpoints/foo.pt
```

which writes it as its own `_myname_embds` weight set. It must share the base
VLM and architecture — a checkpoint on a different backbone needs its own run
with its own `--pretrained_model_path`.

To point the script at different checkpoints:

```bash
bash scripts/get_openvla_vlm_embds.sh \
  --base_vlm_path ./checkpoints/openvla_basevlm_prism-dinosiglip-224px+7b.pt \
  --pretrained_pt_path ./checkpoints/openvla_pretrained_bridge_step-200000-epoch-23-loss=0.0370.pt \
  --carrot_knife_checkpoint ./checkpoints/openvla_cotrained_bridge+sink_carrot-knife_step-005000-epoch-56-loss=0.0368.pt \
  --carrot_knife_vqa_checkpoint ./checkpoints/openvla_vqa_step-005000-epoch-22-loss=0.4131.pt \
  --pot_plate_checkpoint ./checkpoints/openvla_cotrained_bridge+sink_pot-plate_step-005000-epoch-26-loss=0.0030.pt \
  --pot_plate_vqa_checkpoint ./checkpoints/openvla_vqa_pot-plate.pt \
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

OpenVLA additionally gets the two VQA comparisons:

```text
visualization/openvla/pretrained+cotrained_vqa/
visualization/openvla/cotrained+cotrained_vqa/
```

`cotrained+cotrained_vqa/` is the one that isolates what VQA co-training
changes, holding everything else fixed.

Within each comparison directory, plots are split into `all_tasks/` and
per-task directories. Tasks with no embeddings on disk are skipped, so a
partial extraction produces a clean run rather than a wall of warnings.

The `base+pretrained+cotrained/` run is the only one that emits the pairwise-PCA
and discrimination figures, since those need every weight set present at once.

### Raw data

Nothing is figure-only — every number behind a chart is also written to disk:

```text
embeddings/<model>/<task>/<task>_<stem>_<tier>_embds.npz   # the embeddings themselves
visualizations/<model>/<comparison>/all_tasks/
    all_tasks_<emb_type>_cosim.json                 # full cosine-similarity matrix + row labels
    all_tasks_<emb_type>_task_discrimination.json   # separation / discrimination / sensitivity
                                                    # per weight set
```

## 7. Running on SLURM

Both stages have batch scripts under `slurm/`. Submit from the repository root:

```bash
sbatch slurm/extract_openvla_embds.sbatch
```

```bash
sbatch slurm/visualize_openvla_embds.sbatch
```

To chain them so visualization starts only if extraction succeeds:

```bash
jid=$(sbatch --parsable slurm/extract_openvla_embds.sbatch) && sbatch --dependency=afterok:$jid slurm/visualize_openvla_embds.sbatch
```

Extraction requests 1 GPU, 8 CPUs, 128 GB RAM, 3 h on the `gpu` partition. The
memory is host-side, not device-side: the model is ~15 GB in bf16, but each
co-trained `.pt` is ~29 GB of fp32 that `torch.load` pulls into CPU RAM before
the remapped tensors are copied to the GPU. Runtime is dominated by reading
those checkpoints off the network filesystem, not by the 40 forward passes.

Visualization needs no GPU and runs on the `batch` partition.

Logs land in `logs/<job-name>-<job-id>.{out,err}`.

## Notes

- Root helper scripts live in `scripts/`.
- Run scripts from the repository root so relative paths resolve correctly.
- The wrapper scripts bootstrap conda themselves (`scripts/_activate_conda.sh`).
  On this cluster `conda` is a shell function installed by `module load`, and
  functions are not inherited by child processes — so a wrapper invoked as
  `bash scripts/foo.sh`, which is how the SLURM jobs call them, would otherwise
  find no conda at all even when the submitting shell had it.
