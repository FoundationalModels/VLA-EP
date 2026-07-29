#!/usr/bin/env python3
"""Compare two weight sets under three PCA scalings, side by side.

Motivation
----------
A plain PCA scatter of co-trained vs co-trained+VQA is misleading. PC1 carries
~57% of the variance and is dominated by the gap between the two weight sets --
and that gap is largely *magnitude*, not direction: mean embedding norm is ~58
for co-trained versus ~100 for co-trained+VQA. Because the axes are drawn on
their natural scales, PC1 physically stretches the plot, squashing the
within-weight-set structure that the analysis actually cares about. The result
reads as "VQA embeddings are far more spread out" when the cosine metrics say
the opposite (within-task angular spread 0.320 -> 0.166).

Panels
------
1. natural       PCA on raw embeddings, axes on their natural scale.
                 What visualize_vlm_embds.py already produces; kept for reference.

2. equalized     Same PCA, each component divided by its own standard deviation.
                 Both axes then contribute equally to the picture regardless of
                 how much variance they explain, so PC2 structure is visible
                 instead of being flattened by PC1. This is the "weight the axes
                 more evenly" view. Positions are unchanged in rank/ordering --
                 only the aspect ratio of the information changes.

3. unit-norm     PCA refitted on L2-normalized embeddings. Rather than
                 rebalancing the axes after the fact, this removes the magnitude
                 difference at its source, so the projection reflects angular
                 structure only -- the same geometry the cosine-based
                 separation / discrimination / sensitivity metrics operate on.
                 This is the panel to trust when comparing the two weight sets.

Variance percentages are annotated per panel; for the unit-norm panel they come
from the refitted PCA and so differ from the other two.

Usage
-----
  python scripts/plot_weight_set_balanced_pca.py
  python scripts/plot_weight_set_balanced_pca.py --emb_type layer-1_mean
  python scripts/plot_weight_set_balanced_pca.py \
      --weight_sets co-trained co-trained+VQA \
      --output_dir visualizations/openvla/cotrained+cotrained_vqa
"""

import argparse
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from visualize_vlm_embds import (  # noqa: E402
    MODEL_DEFAULTS,
    TASK_COLORS_FT,
    TASK_ORDER,
    TASK_SHORT,
    WEIGHT_SET_ORDER,
    WEIGHT_SET_SIZE,
    WEIGHT_SET_ZORD,
    WS_LEGEND_SHADE,
    _inst_display,
    _ws_edge,
    _ws_legend_label,
    collect_all_tasks_embeddings,
    task_point_color,
)


def _render(ax, X2d, meta, xlabel, ylabel, title, *, annotate=True):
    for i, (task_name, ws, inst_key, _) in enumerate(meta):
        color = task_point_color(task_name, ws)
        marker = "D" if inst_key == "base" else "o"
        edge_c, edge_w = _ws_edge(ws)
        ax.scatter(
            X2d[i, 0], X2d[i, 1],
            c=color, s=WEIGHT_SET_SIZE.get(ws, 110), marker=marker,
            edgecolors=edge_c, linewidths=edge_w,
            zorder=WEIGHT_SET_ZORD.get(ws, 2),
        )
        if annotate:
            ax.annotate(
                f"{TASK_SHORT[task_name]}\n{_inst_display(inst_key)}",
                (X2d[i, 0], X2d[i, 1]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=6, color=color,
            )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    # Equal aspect: with standardized axes, visual distance is only honest if one
    # unit on x is one unit on y.
    ax.set_aspect("equal", adjustable="datalim")


def _legends(ax, meta):
    task_handles = [
        mpatches.Patch(facecolor=TASK_COLORS_FT[t], edgecolor="#555",
                       linewidth=0.6, label=TASK_SHORT[t])
        for t in TASK_ORDER if any(m[0] == t for m in meta)
    ]
    leg1 = ax.legend(handles=task_handles, title="task", fontsize=8,
                     loc="upper left", title_fontsize=8)
    ax.add_artist(leg1)

    ws_present = sorted({m[1] for m in meta},
                        key=lambda w: WEIGHT_SET_ORDER.index(w)
                        if w in WEIGHT_SET_ORDER else 99)
    enc = []
    for ws in ws_present:
        lbl = _ws_legend_label(ws, set(ws_present))
        shade = WS_LEGEND_SHADE.get(ws, "#aaa")
        sz = WEIGHT_SET_SIZE.get(ws, 110)
        edge_c, edge_w = _ws_edge(ws)
        enc.append(plt.scatter([], [], marker="D", c=shade, s=sz,
                               edgecolors=edge_c, linewidths=edge_w,
                               label=f"{lbl} — original"))
        enc.append(plt.scatter([], [], marker="o", c=shade, s=sz,
                               edgecolors=edge_c, linewidths=edge_w,
                               label=f"{lbl} — variant"))
    ax.legend(handles=enc, fontsize=8, loc="lower left")


def main():
    defaults = MODEL_DEFAULTS["openvla"]

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings_dir",
                   default=os.path.join(REPO, "embeddings", defaults["subdir"]))
    p.add_argument("--output_dir",
                   default=os.path.join(REPO, "visualizations", defaults["subdir"],
                                        "cotrained+cotrained_vqa"))
    p.add_argument("--emb_type", default="layer-1_final")
    p.add_argument("--weight_sets", nargs=2,
                   default=["co-trained", "co-trained+VQA"],
                   metavar="WEIGHT_SET")
    p.add_argument("--model", default="openvla", choices=list(MODEL_DEFAULTS))
    args = p.parse_args()

    defaults = MODEL_DEFAULTS[args.model]
    label_to_suffix = {v: k for k, v in defaults["suffix_to_label"].items()}
    try:
        suffixes = [label_to_suffix[w] for w in args.weight_sets]
    except KeyError as e:
        raise SystemExit(
            f"Unknown weight set {e}. Available: {list(label_to_suffix)}") from None

    X, meta = collect_all_tasks_embeddings(
        args.embeddings_dir, args.emb_type, defaults["suffix_to_label"], suffixes)
    if X is None or len(X) < 3:
        raise SystemExit(f"Not enough data in {args.embeddings_dir} for {args.weight_sets}")

    X = X.astype(np.float64)
    print(f"[*] {len(X)} embeddings, dim {X.shape[1]}, weight sets {args.weight_sets}")
    for ws in args.weight_sets:
        norms = [np.linalg.norm(X[i]) for i, m in enumerate(meta) if m[1] == ws]
        print(f"    {ws:<16} n={len(norms):3d}  mean |v| = {np.mean(norms):8.3f}")

    # ── Panel 1 & 2: one PCA, two scalings ──
    pca = PCA(n_components=2)
    X_nat = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    # Equalize: unit variance per component, so neither axis dominates visually.
    X_eq = X_nat / X_nat.std(axis=0, ddof=0)

    # ── Panel 3: magnitude removed before fitting ──
    X_unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    pca_u = PCA(n_components=2)
    X_u2d = pca_u.fit_transform(X_unit)
    var_u = pca_u.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    _render(axes[0], X_nat, meta,
            f"PC1  ({var[0]:.1%} var)", f"PC2  ({var[1]:.1%} var)",
            "1. natural scale\n(PC1 dominates — mostly the magnitude gap)")
    _render(axes[1], X_eq, meta,
            f"PC1 standardized  ({var[0]:.1%} var)",
            f"PC2 standardized  ({var[1]:.1%} var)",
            "2. variance-equalized axes\n(each component scaled to unit variance)")
    _render(axes[2], X_u2d, meta,
            f"PC1  ({var_u[0]:.1%} var)", f"PC2  ({var_u[1]:.1%} var)",
            "3. unit-norm embeddings\n(magnitude removed — angular structure only)")

    _legends(axes[0], meta)

    fig.suptitle(
        f"{defaults['display_name']}  ·  {' vs '.join(args.weight_sets)}  ·  "
        f"{args.emb_type}  ·  PCA under three scalings",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(os.path.join(args.output_dir, "all_tasks"), exist_ok=True)
    stem = f"all_tasks_{args.emb_type}_balanced_pca"
    png = os.path.join(args.output_dir, "all_tasks", f"{stem}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {png}")

    # Persist coordinates and variance ratios, matching the rest of the pipeline.
    payload = {
        "emb_type": args.emb_type,
        "weight_sets": args.weight_sets,
        "explained_variance_ratio": {
            "natural_and_equalized": var.tolist(),
            "unit_norm": var_u.tolist(),
        },
        "mean_norm_by_weight_set": {
            ws: float(np.mean([np.linalg.norm(X[i])
                               for i, m in enumerate(meta) if m[1] == ws]))
            for ws in args.weight_sets
        },
        "points": [
            {
                "task": t, "weight_set": ws, "instruction": ik,
                "natural": X_nat[i].tolist(),
                "equalized": X_eq[i].tolist(),
                "unit_norm": X_u2d[i].tolist(),
            }
            for i, (t, ws, ik, _) in enumerate(meta)
        ],
    }
    js = os.path.join(args.output_dir, "all_tasks", f"{stem}.json")
    with open(js, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved → {js}")


if __name__ == "__main__":
    main()
