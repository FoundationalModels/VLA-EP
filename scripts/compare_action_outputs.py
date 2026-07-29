#!/usr/bin/env python3
"""Compare the ACTIONS each checkpoint predicts, rather than its embeddings.

Why this is a different question
-------------------------------
The embedding analyses measure what a model *represents*. This measures what it
*does*. A representational shift that leaves the predicted action unchanged is
arguably inconsequential, and nothing in the embedding metrics can tell you
which kind you are looking at.

It also sidesteps the shared-basis problem. Embeddings from different lineages
(prism-dinosiglip + Llama-2 vs prism-siglip + Vicuna) live in unrelated spaces,
so cross-model cosines are meaningless and a joint PCA is an artifact. Actions
do not have that problem: every Bridge model emits the same 7-DoF action, so all
checkpoints are directly comparable in one space.

Action space
------------
Token ids are converted with exactly predict_action()'s arithmetic:

    discretized = vocab_size - token_id          (vocab_size = 32000)
    discretized = clip(discretized - 1, 0, 254)
    normalized  = bin_centers[discretized]

with bins = linspace(-1, 1, 256) and bin_centers the 255 midpoints. We stop at
the NORMALIZED action rather than un-normalizing to metres/radians: un-normalizing
needs per-dataset statistics, and the checkpoints disagree about those
(openvla-7b carries stats for many Open-X datasets, the Bridge models only
bridge_orig). Normalized space keeps every model on one scale.

Distances are Euclidean here, not cosine. For a 7-DoF control vector, magnitude
is meaningful -- a small displacement is a different action from a large one in
the same direction -- so cosine would discard exactly what matters. This differs
deliberately from the embedding metrics, where only direction is comparable.

Metrics per checkpoint (mirroring the embedding ones):
  separation      mean Euclidean distance between per-task mean actions
  discrimination  Euclidean silhouette over individual actions, mapped to [0, 1]
  sensitivity     mean pairwise Euclidean distance among instruction variants
  identical_frac  fraction of within-task variant pairs whose 7 action tokens
                  match exactly -- i.e. how often rephrasing changes nothing at
                  all at the level the robot actually executes

Usage:
  python scripts/compare_action_outputs.py
"""

import argparse
import glob
import json
import os
from itertools import combinations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

VOCAB_SIZE = 32000
N_BINS = 256

# Pipeline order, so the table reads in a sensible sequence rather than glob order.
CHECKPOINT_ORDER = ["base", "openx", "bridge-pretrained", "ECoT-steerable",
                    "ECoT-reasoner", "co-trained", "co-trained+VQA"]

METRICS = ["separation", "discrimination", "sensitivity"]

_BINS = np.linspace(-1, 1, N_BINS)
_BIN_CENTERS = (_BINS[:-1] + _BINS[1:]) / 2.0


def tokens_to_action(token_ids):
    """Exactly predict_action()'s de-tokenization, stopping before un-normalization."""
    ids = np.asarray(token_ids)
    discretized = VOCAB_SIZE - ids
    discretized = np.clip(discretized - 1, 0, _BIN_CENTERS.shape[0] - 1)
    return _BIN_CENTERS[discretized]


def metrics_for(actions, task_labels, token_lists):
    per_task = {}
    for a, t in zip(actions, task_labels):
        per_task.setdefault(t, []).append(a)
    per_task = {t: np.stack(v) for t, v in per_task.items()}
    if len(per_task) < 2:
        return None

    cents = np.stack([v.mean(axis=0) for v in per_task.values()])
    n = len(cents)
    separation = float(np.mean([np.linalg.norm(cents[i] - cents[j])
                                for i in range(n) for j in range(n) if i != j]))

    discrimination = None
    if 1 < len(set(task_labels)) < len(task_labels):
        discrimination = 0.5 * (float(silhouette_score(
            np.stack(actions), task_labels, metric="euclidean")) + 1.0)

    sens = [float(np.mean([np.linalg.norm(V[i] - V[j])
                           for i, j in combinations(range(len(V)), 2)]))
            for V in per_task.values() if len(V) > 1]
    sensitivity = float(np.mean(sens)) if sens else None

    # Exact-match rate over within-task variant pairs.
    by_task = {}
    for toks, t in zip(token_lists, task_labels):
        by_task.setdefault(t, []).append(tuple(toks))
    pairs = [(a, b) for v in by_task.values() for a, b in combinations(v, 2)]
    identical = sum(a == b for a, b in pairs) / len(pairs) if pairs else None

    return {"separation": separation, "discrimination": discrimination,
            "sensitivity": sensitivity, "identical_frac": identical,
            "n_actions": len(actions)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--actions_dir", default=os.path.join(REPO, "visualizations", "actions"))
    p.add_argument("--output_dir", default=os.path.join(REPO, "visualizations", "actions"))
    args = p.parse_args()

    files = {}
    for fp in glob.glob(os.path.join(args.actions_dir, "*_language.json")):
        files[os.path.basename(fp)[: -len("_language.json")]] = json.load(open(fp))
    if not files:
        raise SystemExit(f"No *_language.json in {args.actions_dir} -- run the capture first.")

    names = ([n for n in CHECKPOINT_ORDER if n in files]
             + sorted(n for n in files if n not in CHECKPOINT_ORDER))

    rows, per_ckpt_actions = [], {}
    for name in names:
        d = files[name]
        actions, labels, toks = [], [], []
        for task, variants in d["results"].items():
            for inst, r in variants.items():
                ids = r["action_token_ids"]
                if len(ids) != 7:
                    print(f"  [warn] {name}/{task}/{inst}: {len(ids)} action tokens, expected 7")
                    continue
                actions.append(tokens_to_action(ids))
                labels.append(task)
                toks.append(ids)
        if len(actions) < 3:
            print(f"  [skip] {name}: too few valid actions")
            continue
        per_ckpt_actions[name] = (np.stack(actions), labels)
        m = metrics_for(actions, labels, toks)
        if m:
            m["checkpoint"] = name
            rows.append(m)

    if not rows:
        raise SystemExit("No metrics computed.")

    hdr = (f"{'checkpoint':<20}{'n':>4}{'separation':>13}{'discrim':>10}"
           f"{'sensitivity':>13}{'identical':>11}")
    print("\nACTION-SPACE comparison (normalized units, Euclidean)\n")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        f = lambda v: f"{v:.5f}" if v is not None else "n/a"  # noqa: E731
        ident = "n/a" if r["identical_frac"] is None else f"{r['identical_frac']:.0%}"
        print(f"{r['checkpoint']:<20}{r['n_actions']:>4}{f(r['separation']):>13}"
              f"{f(r['discrimination']):>10}{f(r['sensitivity']):>13}{ident:>11}")

    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(6 * 4, 5.4))
    labels_x = [r["checkpoint"] for r in rows]
    x = np.arange(len(rows))
    for ax, metric in zip(axes, METRICS + ["identical_frac"]):
        vals = [r[metric] if r[metric] is not None else np.nan for r in rows]
        ax.bar(x, vals, color="#0173b2", edgecolor="#333", linewidth=0.6)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels_x, fontsize=7, rotation=30, ha="right")
        title = {"separation": "between-task separation",
                 "discrimination": "between-task discrimination",
                 "sensitivity": "within-task instruction sensitivity",
                 "identical_frac": "variant pairs with IDENTICAL action"}[metric]
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    axes[0].set_ylabel("normalized action units (Euclidean)")
    axes[3].set_ylabel("fraction")

    fig.suptitle("Predicted ACTION comparison  ·  7-DoF normalized action space  ·  "
                 "all checkpoints directly comparable (no shared-basis caveat)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = os.path.join(args.output_dir, "action_metric_comparison.png")
    fig.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved → {png}")

    js = os.path.join(args.output_dir, "action_metric_comparison.json")
    with open(js, "w") as f:
        json.dump({"space": "normalized action (bin centers, [-1,1]), Euclidean",
                   "rows": rows}, f, indent=2)
    print(f"Saved → {js}")


if __name__ == "__main__":
    main()
