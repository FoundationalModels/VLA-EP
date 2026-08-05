#!/usr/bin/env python3
"""Standalone slide figures for the three referring-expression metrics.

  sensitivity     mean cosine distance between the five wordings of one task.
                  LOWER is better: all five denote the same thing.
  separation      mean cosine distance between the two task centroids.
                  HIGHER is better: the policy must tell the tasks apart.
  discrimination  cosine silhouette over individual embeddings, mapped to [0,1]
                  with 0.5 = chance. HIGHER is better. This is a RATIO of
                  between-task distance to within-task spread, which is why it
                  behaves differently from the other two.

Error bars are the range from recomputing with each wording left out in turn
(5 recomputations, 4 wordings each). They answer "is this difference real, or an
artifact of which five sentences the benchmark happens to use?" Read them before
reading the ranking: on this sample size the discrimination bars overlap almost
completely, so its ordering is not interpretable, whereas sensitivity separates
cleanly. The bars are the point of these figures as much as the values are.

Bars are sorted best-first for whichever metric is plotted. The untrained base VLM
is drawn in grey as the reference point -- it is not a policy, it is where the
model starts before any robot data.

Usage:
  python scripts/plot_sensitivity_figure.py                      # sensitivity
  python scripts/plot_sensitivity_figure.py --metric separation
  python scripts/plot_sensitivity_figure.py --metric discrimination
"""

import argparse
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

from visualize_vlm_embds import MODEL_DEFAULTS, collect_task_group_embeddings  # noqa: E402

TASKS = ("put_carrot_on_plate", "put_knife_on_plate")
INST = ["base", "S-PROP", "S-LANG", "S-MO", "S-INT"]

# Display-name overrides ONLY. The set of bars is determined by what is on disk in
# embeddings/openvla/, not by this dict -- a weight set missing here still gets a
# bar, labelled with its raw internal key. Keys are the labels defined in
# visualize_vlm_embds.MODEL_DEFAULTS["openvla"]["suffix_to_label"].
DISPLAY = {
    "base":              "Base VLM\n(no robot data)",
    "Open-X":            "OXE PT",
    "Open-X+FT":         "OXE FT",
    "bridge-pretrained": "Bridge PT",
    "vqa-pretrained":    "Bridge+VQA PT",
    "co-trained":        "Bridge FT",
    "co-trained+VQA":    "Bridge+VQA FT",
    "ECoT-steerable":    "Bridge Steerable",
    "ECoT-reasoner":     "Bridge Steerable-ER",
}


def sensitivity(X, meta, idx):
    """Mean cosine distance among the wordings of one task, averaged over tasks."""
    out = []
    for t in TASKS:
        V = X[[i for i in idx if meta[i][0] == t]]
        if len(V) > 1:
            s = cosine_similarity(V)
            out.append(np.mean([1 - s[i, j] for i, j in combinations(range(len(V)), 2)]))
    return float(np.mean(out))


def separation(X, meta, idx):
    """Mean cosine distance between task centroids."""
    cents = [X[[i for i in idx if meta[i][0] == t]].mean(axis=0) for t in TASKS
             if any(meta[i][0] == t for i in idx)]
    if len(cents) < 2:
        return float("nan")
    s = cosine_similarity(np.stack(cents))
    n = len(cents)
    return float(np.mean([1 - s[i, j] for i in range(n) for j in range(n) if i != j]))


def discrimination(X, meta, idx):
    """Cosine silhouette over individual embeddings, mapped to [0,1] (0.5 = chance)."""
    labels = [meta[i][0] for i in idx]
    if not (1 < len(set(labels)) < len(labels)):
        return float("nan")
    return 0.5 * (float(silhouette_score(X[idx], labels, metric="cosine")) + 1.0)


# metric -> (function, higher_is_better, axis label, title)
METRICS = {
    "sensitivity": (sensitivity, False,
                    "Within-task instruction sensitivity  (cosine distance)   —   lower is better",
                    "How far apart are five ways of saying the same instruction?"),
    "separation": (separation, True,
                   "Between-task separation  (cosine distance)   —   higher is better",
                   "How far apart are the two tasks?"),
    "discrimination": (discrimination, True,
                       "Between-task discrimination  (silhouette, 0.5 = chance)   —   higher is better",
                       "Are the tasks separable relative to their own spread?"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb_type", default="layer-1_final")
    p.add_argument("--metric", default="sensitivity", choices=list(METRICS))
    p.add_argument("--output", default=None,
                   help="Defaults to visualizations/comparison/{metric}_figure.png")
    args = p.parse_args()

    fn, higher_better, xlabel, title = METRICS[args.metric]
    if args.output is None:
        args.output = os.path.join(REPO, "visualizations", "comparison",
                                   f"{args.metric}_figure.png")

    d = MODEL_DEFAULTS["openvla"]
    X, meta = collect_task_group_embeddings(os.path.join(REPO, "embeddings", "openvla"),
                                            TASKS, args.emb_type, d["suffix_to_label"])
    X = X.astype(np.float64)

    rows = []
    for ws in dict.fromkeys(m[1] for m in meta):
        idx = [i for i, m in enumerate(meta) if m[1] == ws]
        full = fn(X, meta, idx)
        drops = [fn(X, meta, [i for i in idx if meta[i][2] != dr]) for dr in INST]
        rows.append({"ws": ws, "value": full, "lo": min(drops), "hi": max(drops)})

    rows.sort(key=lambda r: r["value"], reverse=higher_better)

    # Warn rather than silently mixing prettified labels with raw internal keys,
    # which is what happens when a new weight set is added upstream.
    missing = [r["ws"] for r in rows if r["ws"] not in DISPLAY]
    if missing:
        print(f"  [warn] no DISPLAY name for {missing} -- using the raw key(s); "
              f"add them to DISPLAY to control the label")

    labels = [DISPLAY.get(r["ws"], r["ws"]) for r in rows]
    vals = [r["value"] for r in rows]
    lo = [r["value"] - r["lo"] for r in rows]
    hi = [r["hi"] - r["value"] for r in rows]
    # Base is the pre-robot reference, not a policy: grey it out.
    colors = ["#999999" if r["ws"] == "base" else "#0173b2" for r in rows]

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(y, vals, xerr=[lo, hi], color=colors, edgecolor="#222", linewidth=0.8,
            error_kw=dict(ecolor="#444", capsize=4, lw=1.2), zorder=3)

    span = max(r["hi"] for r in rows) - min(r["lo"] for r in rows)
    for yi, r in zip(y, rows):
        ax.text(r["hi"] + span * 0.04, yi, f"{r['value']:.3f}", va="center", ha="left",
                fontsize=11, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()   # best at top
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    if args.metric == "discrimination":
        lo_ax = min(r["lo"] for r in rows); hi_ax = max(r["hi"] for r in rows)
        pad = (hi_ax - lo_ax) * 0.30
        ax.set_xlim(lo_ax - pad, hi_ax + pad * 1.6)
        ax.axvline(0.5, color="#b22222", lw=1.2, ls=":", zorder=2)
        ax.text(0.5, -0.9, " chance", color="#b22222", fontsize=9, va="top")
    else:
        ax.set_xlim(0, max(r["hi"] for r in rows) * 1.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.text(0.5, -0.02,
             "Error bars: range across recomputing with each of the five wordings left out in turn.",
             ha="center", fontsize=9, color="#555")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")
    for r in rows:
        print(f"  {r['ws']:<20}{r['value']:.3f}   [{r['lo']:.3f} – {r['hi']:.3f}]")


if __name__ == "__main__":
    main()
