#!/usr/bin/env python3
"""Plot checkpoint-group discrimination/sensitivity summary lines.

For each model, compute two mean cosine-distance metrics per checkpoint group:
  - between-task discrimination: cosine distance between task centroids
  - within-task sensitivity: mean pairwise cosine distance among instruction variants

The x-axis is the model weight stage (base, pre-trained, co-trained).  Each model
panel contains four lines: two checkpoint groups × two metrics.
"""

from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from visualize_vlm_embds import (  # noqa: E402
    CHECKPOINT_TASK_GROUPS,
    MODEL_DEFAULTS,
    WEIGHT_SET_ORDER,
    collect_task_group_embeddings,
)


VIZ_ROOT = REPO / "visualizations"

MODEL_EMB_TYPES = {
    "pi0": "last_layer",
    "openvla": "layer-1_final",
    "minivla": "layer-1_final",
}

MODEL_LABELS = {
    "pi0": "Pi0 - last layer KV cache",
    "openvla": "OpenVLA - last layer last token",
    "minivla": "MiniVLA - last layer last token",
}

# Seaborn "colorblind" palette, dark variants.
GROUP_STYLES = {
    "carrot_knife": ("#0173b2", "carrot/knife"),
    "pot_plate": ("#de8f05", "pot/plate"),
}

METRIC_STYLES = {
    "between": ("o", "-", "between-task discrimination"),
    "within": ("s", "--", "within-task sensitivity"),
}


def _between_task_distance(X, meta, task_names, weight_set):
    task_means = []
    for task_name in task_names:
        mask = [i for i, m in enumerate(meta) if m[0] == task_name and m[1] == weight_set]
        if mask:
            task_means.append(X[mask].mean(axis=0))

    if len(task_means) < 2:
        return np.nan

    sim = cosine_similarity(np.stack(task_means))
    n = sim.shape[0]
    vals = [1.0 - sim[i, j] for i in range(n) for j in range(n) if i != j]
    return float(np.mean(vals))


def _within_task_sensitivity(X, meta, task_names, weight_set):
    per_task = []
    for task_name in task_names:
        mask = [i for i, m in enumerate(meta) if m[0] == task_name and m[1] == weight_set]
        if len(mask) < 2:
            continue

        sim = cosine_similarity(X[mask])
        n = sim.shape[0]
        vals = [1.0 - sim[i, j] for i in range(n) for j in range(n) if i != j]
        per_task.append(float(np.mean(vals)))

    return float(np.mean(per_task)) if per_task else np.nan


def _model_metrics(model_name):
    defaults = MODEL_DEFAULTS[model_name]
    emb_type = MODEL_EMB_TYPES[model_name]
    embeddings_dir = REPO / "embeddings" / defaults["subdir"]
    suffix_to_label = defaults["suffix_to_label"]

    metrics = {}
    for group_key, task_names, _ in CHECKPOINT_TASK_GROUPS:
        X, meta = collect_task_group_embeddings(
            str(embeddings_dir), task_names, emb_type, suffix_to_label)
        if X is None or len(X) < 2:
            continue

        metrics[group_key] = {
            "between": [
                _between_task_distance(X, meta, task_names, ws)
                for ws in WEIGHT_SET_ORDER
            ],
            "within": [
                _within_task_sensitivity(X, meta, task_names, ws)
                for ws in WEIGHT_SET_ORDER
            ],
        }

    return emb_type, metrics


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    x = np.arange(len(WEIGHT_SET_ORDER))

    for ax, model_name in zip(axes, ("pi0", "openvla", "minivla")):
        emb_type, metrics = _model_metrics(model_name)

        for group_key, group_metrics in metrics.items():
            color, group_label = GROUP_STYLES[group_key]
            for metric_key, ys in group_metrics.items():
                marker, linestyle, metric_label = METRIC_STYLES[metric_key]
                ax.plot(
                    x, ys,
                    color=color, linestyle=linestyle, marker=marker,
                    linewidth=2.0, markersize=6,
                    label=f"{group_label} · {metric_label}",
                )

        ax.set_title(MODEL_LABELS[model_name], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(WEIGHT_SET_ORDER)
        ax.set_ylabel("mean cosine distance")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Task discrimination and instruction sensitivity across weight stages",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))

    out = VIZ_ROOT / "summary_discrimination_sensitivity_lines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")

    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_pdf}")


if __name__ == "__main__":
    main()
