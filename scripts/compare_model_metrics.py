#!/usr/bin/env python3
"""Tabulate separation / discrimination / sensitivity across models and weight sets.

This is the comparison that stays valid across architectures. All three metrics
are cosine-based and computed entirely within a single model's own embedding
space, so they carry no dependence on a shared basis -- unlike a joint PCA or a
cross-model cosine heatmap, which require the spaces to be aligned and are
meaningless when they are not.

That matters here because the models involved are not all the same lineage:

  openvla family   prism-dinosiglip-224px+7b, Llama-2-7B, "In:/Out:" prompt
    base / pre-trained / steerable / co-trained / co-trained+VQA
  ecot             prism-siglip, Vicuna-7B-v1.5, "USER:/ASSISTANT:" prompt

The OpenVLA weight sets share one space and are directly comparable point by
point. ECoT is only comparable through these summary metrics, and even then its
numbers carry a prompt-template difference alongside the architecture one --
each model is extracted with the template it was trained for, which is correct
per model but is one more thing that differs between them.

Metric definitions match visualize_vlm_embds.make_task_discrimination_chart:
  separation      mean cosine distance between task centroids (higher = tasks
                  further apart)
  discrimination  cosine silhouette over individual embeddings, mapped to [0, 1]
                  (higher = task clusters cleaner relative to their own spread)
  sensitivity     mean pairwise cosine distance among instruction variants of the
                  same task (higher = embedding moves more under rephrasing)

Usage:
  python scripts/compare_model_metrics.py
  python scripts/compare_model_metrics.py --emb_type layer-1_mean
"""

import argparse
import json
import os
import sys
from itertools import combinations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from visualize_vlm_embds import (  # noqa: E402
    MODEL_DEFAULTS,
    WEIGHT_SET_ORDER,
    collect_task_group_embeddings,
)

# (model key, embeddings subdir) -- subdir overrides MODEL_DEFAULTS when the same
# model was extracted more than once (e.g. ECoT under two prompt templates).
DEFAULT_SOURCES = [
    ("openvla", "openvla"),
    ("ecot", "ecot"),
]

TASK_GROUP = ("put_carrot_on_plate", "put_knife_on_plate")

METRICS = ["separation", "discrimination", "sensitivity"]


def compute_metrics(X, meta, task_names, weight_set):
    """Return {metric: value} for one weight set, or None if too little data."""
    idx = [i for i, m in enumerate(meta) if m[1] == weight_set]
    if len(idx) < 2:
        return None

    per_task = {}
    for t in task_names:
        mask = [i for i in idx if meta[i][0] == t]
        if mask:
            per_task[t] = X[mask]
    if len(per_task) < 2:
        return None

    cents = np.stack([v.mean(axis=0) for v in per_task.values()])
    sim = cosine_similarity(cents)
    n = len(cents)
    separation = 1.0 - float(np.mean([sim[i, j]
                                      for i in range(n) for j in range(n) if i != j]))

    labels = [meta[i][0] for i in idx]
    discrimination = None
    if 1 < len(set(labels)) < len(labels):
        discrimination = 0.5 * (float(silhouette_score(
            X[idx], labels, metric="cosine")) + 1.0)

    sens = []
    for V in per_task.values():
        if len(V) < 2:
            continue
        s = cosine_similarity(V)
        sens.append(float(np.mean([1.0 - s[i, j]
                                   for i, j in combinations(range(len(V)), 2)])))
    sensitivity = float(np.mean(sens)) if sens else None

    return {
        "separation": separation,
        "discrimination": discrimination,
        "sensitivity": sensitivity,
        "n_embeddings": len(idx),
        "mean_norm": float(np.linalg.norm(X[idx], axis=1).mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb_type", default="layer-1_final")
    p.add_argument("--output_dir", default=os.path.join(REPO, "visualizations", "comparison"))
    p.add_argument("--sources", nargs="+", default=None, metavar="MODEL[:SUBDIR]",
                   help="Models to include, optionally with an embeddings subdir "
                        "override, e.g. 'ecot:ecot_plainprompt'.")
    p.add_argument("--sort_by", default=None, choices=METRICS,
                   help="Order the bars by this metric (descending) instead of by "
                        "pipeline stage. All three panels keep the same order, so a "
                        "metric that does not track the sort key is immediately "
                        "visible as a non-monotonic panel. Recommended: "
                        "discrimination, the only one of the three carrying "
                        "independent signal.")
    p.add_argument("--ascending", action="store_true",
                   help="With --sort_by, order smallest-first instead of largest-first.")
    p.add_argument("--suffix", default="",
                   help="Appended to the output filenames, to keep a filtered or "
                        "re-sorted run from overwriting the full comparison.")
    args = p.parse_args()

    if args.sources:
        sources = []
        for s in args.sources:
            model, _, sub = s.partition(":")
            sources.append((model, sub or MODEL_DEFAULTS[model]["subdir"]))
    else:
        sources = DEFAULT_SOURCES

    rows = []
    for model, subdir in sources:
        defaults = MODEL_DEFAULTS[model]
        emb_dir = os.path.join(REPO, "embeddings", subdir)
        if not os.path.isdir(emb_dir):
            print(f"[skip] {model}: no embeddings at {emb_dir}")
            continue

        X, meta = collect_task_group_embeddings(
            emb_dir, TASK_GROUP, args.emb_type, defaults["suffix_to_label"])
        if X is None:
            print(f"[skip] {model}: no embeddings for {args.emb_type}")
            continue
        X = X.astype(np.float64)

        present = [ws for ws in WEIGHT_SET_ORDER if any(m[1] == ws for m in meta)]
        # Weight sets that are not part of the shared pipeline ordering (e.g. the
        # single "ECoT" set) still need to appear.
        present += [ws for ws in dict.fromkeys(m[1] for m in meta) if ws not in present]

        for ws in present:
            r = compute_metrics(X, meta, TASK_GROUP, ws)
            if r is None:
                continue
            # Distinguish rows that share a model+weight_set but came from
            # different extractions (e.g. ECoT under two prompt templates).
            variant = "" if subdir == defaults["subdir"] else f" [{subdir}]"
            r.update(model=model, subdir=subdir, weight_set=ws,
                     display=f"{ws}{variant}")
            rows.append(r)

    if not rows:
        raise SystemExit("No metrics computed -- have the extractions run?")

    if args.sort_by:
        # None sorts last regardless of direction, so a missing value never
        # displaces a real one from the top of the chart.
        rows.sort(key=lambda r: (r[args.sort_by] is None,
                                 r[args.sort_by] if r[args.sort_by] is not None else 0),
                  reverse=not args.ascending)
        # reverse=True would also flip the None-sorts-last rule, so re-append them.
        if not args.ascending:
            missing = [r for r in rows if r[args.sort_by] is None]
            rows = [r for r in rows if r[args.sort_by] is not None] + missing

    # ── Table ──
    hdr = f"{'model':<10}{'weight set':<30}{'n':>4}{'mean|v|':>10}" \
          f"{'separation':>13}{'discrim':>10}{'sensitivity':>13}"
    print(f"\nembedding type: {args.emb_type}   tasks: {', '.join(TASK_GROUP)}\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f = lambda v: f"{v:.5f}" if v is not None else "n/a"  # noqa: E731
        print(f"{r['model']:<10}{r['display']:<30}{r['n_embeddings']:>4}"
              f"{r['mean_norm']:>10.2f}{f(r['separation']):>13}"
              f"{f(r['discrimination']):>10}{f(r['sensitivity']):>13}")

    # ── Grouped bar chart ──
    os.makedirs(args.output_dir, exist_ok=True)
    labels = [f"{r['model']}\n{r['display']}" for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5.2))
    for ax, metric in zip(axes, METRICS):
        vals = [r[metric] if r[metric] is not None else np.nan for r in rows]
        colors = ["#0173b2" if r["model"] == "openvla" else "#de8f05" for r in rows]
        is_key = metric == args.sort_by
        ax.bar(x, vals, color=colors,
               edgecolor="#000" if is_key else "#333",
               linewidth=1.6 if is_key else 0.6)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        if is_key:
            title = f"{metric}\n(sorted by this)"
        elif args.sort_by:
            # Monotonic here too would mean this metric is redundant with the key.
            clean = [v for v in vals if not np.isnan(v)]
            mono = (all(a >= b for a, b in zip(clean, clean[1:]))
                    or all(a <= b for a, b in zip(clean, clean[1:])))
            title = f"{metric}\n({'tracks the sort key' if mono else 'independent of sort key'})"
        else:
            title = metric
        ax.set_title(title, fontsize=11, fontweight="bold" if is_key else "normal")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("cosine-based value")

    subtitle = ("blue = OpenVLA family (shared space) · orange = OpenVLA-v0.1 / ECoT "
                "(separate architecture, comparable via these metrics only)")
    if len({r["model"] for r in rows}) == 1:
        subtitle = f"{rows[0]['model']} family"
    if args.sort_by:
        subtitle += (f"  ·  bars ordered by {args.sort_by}, "
                     f"{'ascending' if args.ascending else 'descending'}")
    fig.suptitle(
        f"Task separation / discrimination / instruction sensitivity  ·  {args.emb_type}\n"
        + subtitle,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    tag = f"{args.emb_type}{args.suffix}"
    if args.sort_by:
        tag += f"_by_{args.sort_by}"
    png = os.path.join(args.output_dir, f"model_metric_comparison_{tag}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {png}")

    js = os.path.join(args.output_dir, f"model_metric_comparison_{tag}.json")
    with open(js, "w") as f:
        json.dump({"emb_type": args.emb_type, "tasks": list(TASK_GROUP), "rows": rows},
                  f, indent=2)
    print(f"Saved → {js}")


if __name__ == "__main__":
    main()
