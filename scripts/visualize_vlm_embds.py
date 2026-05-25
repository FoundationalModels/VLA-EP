"""
Visualize Pi0 VLM embeddings for a given task or all tasks together.

Per-task mode (positional task_name argument):
  Produces three plots per embedding type (all_layers / last_layer):
    - T-SNE scatter
    - PCA scatter  (PC axes show % variance explained)
    - Cosine similarity heatmap
  Saved to visualization/{task_name}_{emb_type}_{method}.png

All-tasks mode (no positional argument):
  Same three plot types, combining all 4 tasks in one figure.
  Color encodes task identity; marker/shade encodes pretrained vs finetuned.
  Saved to visualization/all_tasks_{emb_type}_{method}.png

Color scheme (scatter plots):
  post-trained  base instruction  → task color, diamond marker
  post-trained  other variants    → task color, circle marker
  pre-trained   (any)             → faded task color
"""

import argparse
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# ── Single-task color scheme ───────────────────────────────────────────────────
C_FT_BASE  = "#2ecc71"
C_FT_OTHER = "#2980b9"

def _fade(hex_color: str, alpha: float = 0.55) -> str:
    """Blend hex_color with white at (1-alpha) weight."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * alpha + 255 * (1 - alpha)),
        int(g * alpha + 255 * (1 - alpha)),
        int(b * alpha + 255 * (1 - alpha)),
    )

C_PT_BASE  = _fade(C_FT_BASE)
C_PT_OTHER = _fade(C_FT_OTHER)

INST_ORDER = ["base", "S-PROP", "S-LANG", "S-MO", "S-INT"]

# ── All-tasks color scheme ─────────────────────────────────────────────────────
TASK_ORDER = [
    "put_carrot_on_plate",
    "put_knife_on_plate",
    "flip_pot_upright",
    "put_plate_in_sink",
]

TASK_COLORS_FT = {
    "put_carrot_on_plate": "#e74c3c",   # red
    "put_knife_on_plate":  "#e67e22",   # orange
    "flip_pot_upright":    "#9b59b6",   # purple
    "put_plate_in_sink":   "#1abc9c",   # teal
}

TASK_SHORT = {
    "put_carrot_on_plate": "carrot",
    "put_knife_on_plate":  "knife",
    "flip_pot_upright":    "pot",
    "put_plate_in_sink":   "plate",
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def find_embedding_files(embeddings_dir: str, task_name: str):
    task_dir = os.path.join(embeddings_dir, task_name)
    return sorted(glob.glob(os.path.join(task_dir, f"{task_name}_*.npz")))


def parse_file_info(filepath: str, task_name: str):
    """Return (is_pretrained: bool, model_label: str)."""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    body = stem[len(task_name) + 1:]
    if body.endswith("_pretrained_embds"):
        return True, body[: -len("_pretrained_embds")]
    if body.endswith("_finetuned_embds"):
        return False, body[: -len("_finetuned_embds")]
    return None, body


def load_vecs(npz_path: str, emb_type: str) -> dict:
    data = np.load(npz_path)
    return {
        key: data[f"{key}_{emb_type}"][0]
        for key in INST_ORDER
        if f"{key}_{emb_type}" in data
    }


def collect_embeddings(embeddings_dir: str, task_name: str, emb_type: str):
    """
    Returns
    -------
    X    : np.ndarray [n, dim]
    meta : list of (is_pretrained: bool, inst_key: str, model_label: str)
    """
    files = find_embedding_files(embeddings_dir, task_name)
    vecs, meta = [], []
    for fp in files:
        is_pt, model_label = parse_file_info(fp, task_name)
        for inst_key, vec in load_vecs(fp, emb_type).items():
            vecs.append(vec)
            meta.append((is_pt, inst_key, model_label))
    if not vecs:
        return None, None
    return np.stack(vecs), meta


def collect_all_tasks_embeddings(embeddings_dir: str, emb_type: str):
    """
    Returns
    -------
    X    : np.ndarray [n, dim]
    meta : list of (task_name: str, is_pretrained: bool, inst_key: str, model_label: str)
    """
    vecs, meta = [], []
    for task_name in TASK_ORDER:
        files = find_embedding_files(embeddings_dir, task_name)
        if not files:
            print(f"  [warn] no embedding files for task: {task_name}")
            continue
        # Sort: pretrained first, then finetuned
        pt_files = [f for f in files if f.endswith("_pretrained_embds.npz")]
        ft_files = [f for f in files if f.endswith("_finetuned_embds.npz")]
        for fp in pt_files + ft_files:
            is_pt, model_label = parse_file_info(fp, task_name)
            for inst_key, vec in load_vecs(fp, emb_type).items():
                vecs.append(vec)
                meta.append((task_name, is_pt, inst_key, model_label))
    if not vecs:
        return None, None
    return np.stack(vecs), meta


def point_color(is_pretrained: bool, is_base: bool) -> str:
    if is_pretrained:
        return C_PT_BASE if is_base else C_PT_OTHER
    return C_FT_BASE if is_base else C_FT_OTHER


def task_point_color(task_name: str, is_pretrained: bool) -> str:
    base_color = TASK_COLORS_FT[task_name]
    return _fade(base_color, alpha=0.45) if is_pretrained else base_color


# ── Single-task scatter renderer ──────────────────────────────────────────────

def _render_scatter(ax, X2d, meta, xlabel: str, ylabel: str, title: str) -> None:
    used = set()
    for i, (is_pt, inst_key, _) in enumerate(meta):
        is_base = inst_key == "base"
        color = point_color(is_pt, is_base)
        ax.scatter(
            X2d[i, 0], X2d[i, 1],
            c=color, s=150 if not is_pt else 120,
            marker="o", edgecolors="#555", linewidths=0.6,
            zorder=3 if not is_pt else 2,
        )
        ax.annotate(
            inst_key, (X2d[i, 0], X2d[i, 1]),
            textcoords="offset points", xytext=(6, 5),
            fontsize=8, color=color,
            fontweight="bold" if not is_pt else "normal",
        )
        used.add(("pretrained" if is_pt else "finetuned", "base" if is_base else "other"))

    legend_spec = [
        (("finetuned",  "base"),  C_FT_BASE,  "post-trained — base"),
        (("finetuned",  "other"), C_FT_OTHER, "post-trained — variant"),
        (("pretrained", "base"),  C_PT_BASE,  "pre-trained — base"),
        (("pretrained", "other"), C_PT_OTHER, "pre-trained — variant"),
    ]
    handles = [
        mpatches.Patch(facecolor=c, edgecolor="#555", linewidth=0.6, label=lbl)
        for key, c, lbl in legend_spec if key in used
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")


# ── All-tasks scatter renderer ────────────────────────────────────────────────

def _render_all_tasks_scatter(ax, X2d, meta, xlabel: str, ylabel: str, title: str) -> None:
    for i, (task_name, is_pt, inst_key, _) in enumerate(meta):
        color = task_point_color(task_name, is_pt)
        marker = "D" if inst_key == "base" else "o"
        size = 130 if not is_pt else 90
        ax.scatter(
            X2d[i, 0], X2d[i, 1],
            c=color, s=size, marker=marker,
            edgecolors="#555", linewidths=0.5,
            zorder=3 if not is_pt else 2,
        )
        ax.annotate(
            f"{TASK_SHORT[task_name]}\n{inst_key}",
            (X2d[i, 0], X2d[i, 1]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=6, color=color,
            fontweight="bold" if not is_pt else "normal",
        )

    # Legend: task colors
    task_handles = [
        mpatches.Patch(
            facecolor=TASK_COLORS_FT[t], edgecolor="#555", linewidth=0.6,
            label=TASK_SHORT[t],
        )
        for t in TASK_ORDER if any(m[0] == t for m in meta)
    ]
    # Legend: shade/marker encoding
    enc_handles = [
        plt.scatter([], [], marker="D", c="#888", s=100, edgecolors="#555",
                    linewidths=0.5, label="post-trained — base"),
        plt.scatter([], [], marker="o", c="#888", s=100, edgecolors="#555",
                    linewidths=0.5, label="post-trained — variant"),
        plt.scatter([], [], marker="o", c="#ccc", s=70, edgecolors="#555",
                    linewidths=0.5, label="pre-trained (faded)"),
    ]
    leg1 = ax.legend(handles=task_handles, title="task", fontsize=8,
                     loc="upper left", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=enc_handles, fontsize=8, loc="lower left")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")


# ── T-SNE ─────────────────────────────────────────────────────────────────────

def make_tsne_plot(embeddings_dir, task_name, output_dir, emb_type) -> None:
    X, meta = collect_embeddings(embeddings_dir, task_name, emb_type)
    if X is None or len(X) < 2:
        print(f"  [T-SNE] not enough data"); return

    X2d = TSNE(
        n_components=2, perplexity=min(5, len(X) - 1),
        random_state=42, max_iter=2000, init="pca",
    ).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    _render_scatter(ax, X2d, meta,
                    xlabel="T-SNE 1", ylabel="T-SNE 2",
                    title=f"{task_name}  ·  {emb_type}  T-SNE")
    _save(fig, output_dir, f"{task_name}_{emb_type}_tsne.png", task_name=task_name)


def make_all_tasks_tsne_plot(embeddings_dir, output_dir, emb_type) -> None:
    X, meta = collect_all_tasks_embeddings(embeddings_dir, emb_type)
    if X is None or len(X) < 2:
        print(f"  [T-SNE all-tasks] not enough data"); return

    perplexity = min(15, len(X) - 1)
    X2d = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=42, max_iter=2000, init="pca",
    ).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    _render_all_tasks_scatter(ax, X2d, meta,
                              xlabel="T-SNE 1", ylabel="T-SNE 2",
                              title=f"all tasks  ·  {emb_type}  T-SNE")
    _save(fig, output_dir, f"all_tasks_{emb_type}_tsne.png", task_name="all_tasks")


# ── PCA ───────────────────────────────────────────────────────────────────────

def make_pca_plot(embeddings_dir, task_name, output_dir, emb_type) -> None:
    X, meta = collect_embeddings(embeddings_dir, task_name, emb_type)
    if X is None or len(X) < 2:
        print(f"  [PCA] not enough data"); return

    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    _render_scatter(ax, X2d, meta,
                    xlabel=f"PC1  ({var[0]:.1%} var)",
                    ylabel=f"PC2  ({var[1]:.1%} var)",
                    title=f"{task_name}  ·  {emb_type}  PCA")
    _save(fig, output_dir, f"{task_name}_{emb_type}_pca.png", task_name=task_name)


def make_all_tasks_pca_plot(embeddings_dir, output_dir, emb_type) -> None:
    X, meta = collect_all_tasks_embeddings(embeddings_dir, emb_type)
    if X is None or len(X) < 2:
        print(f"  [PCA all-tasks] not enough data"); return

    pca = PCA(n_components=2)
    X2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    _render_all_tasks_scatter(ax, X2d, meta,
                              xlabel=f"PC1  ({var[0]:.1%} var)",
                              ylabel=f"PC2  ({var[1]:.1%} var)",
                              title=f"all tasks  ·  {emb_type}  PCA")
    _save(fig, output_dir, f"all_tasks_{emb_type}_pca.png", task_name="all_tasks")


# ── Cosine similarity heatmap ─────────────────────────────────────────────────

def make_cosim_plot(embeddings_dir, task_name, output_dir, emb_type) -> None:
    X, meta = collect_embeddings(embeddings_dir, task_name, emb_type)
    if X is None or len(X) < 2:
        print(f"  [CosSim] not enough data"); return

    sim = cosine_similarity(X)   # [n, n]
    n = len(meta)

    # Group by file (pretrained first, then finetuned by model_label)
    seen = []
    for is_pt, _, ml in meta:
        if (is_pt, ml) not in seen:
            seen.append((is_pt, ml))
    seen.sort(key=lambda x: (0 if x[0] else 1, x[1]))

    ordered_idx = []
    for (is_pt, ml) in seen:
        for inst_key in INST_ORDER:
            for i, (ipt, ik, iml) in enumerate(meta):
                if ipt == is_pt and ik == inst_key and iml == ml:
                    ordered_idx.append(i)

    sim_ord = sim[np.ix_(ordered_idx, ordered_idx)]
    meta_ord = [meta[i] for i in ordered_idx]

    multi_model = len(seen) > 1

    def short_label(is_pt, inst_key, model_label):
        if not multi_model:
            return inst_key
        prefix = "PT" if is_pt else model_label.split("_")[0]
        return f"{prefix}\n{inst_key}"

    tick_labels = [short_label(*m) for m in meta_ord]
    tick_colors = [point_color(m[0], m[1] == "base") for m in meta_ord]

    group_sizes = []
    for grp in seen:
        count = sum(1 for m in meta_ord if (m[0], m[2]) == grp)
        group_sizes.append(count)
    separators = np.cumsum(group_sizes)[:-1]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.85), max(5, n * 0.75)))
    im = ax.imshow(sim_ord, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="cosine similarity", fraction=0.046, pad=0.04)

    for r in range(n):
        for c in range(n):
            val = sim_ord[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.35 < val < 0.85 else "white")

    for sep in separators:
        ax.axhline(sep - 0.5, color="white", linewidth=2)
        ax.axvline(sep - 0.5, color="white", linewidth=2)

    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=7)

    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(color)

    ax.set_title(f"{task_name}  ·  {emb_type}  cosine similarity", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, f"{task_name}_{emb_type}_cosim.png", task_name=task_name)


def make_all_tasks_cosim_plot(embeddings_dir, output_dir, emb_type) -> None:
    X, meta = collect_all_tasks_embeddings(embeddings_dir, emb_type)
    if X is None or len(X) < 2:
        print(f"  [CosSim all-tasks] not enough data"); return

    n = len(meta)
    sim = cosine_similarity(X)   # [n, n]  (already in TASK_ORDER × PT-first order)

    # Build tick labels and colors
    tick_labels, tick_colors = [], []
    for task_name, is_pt, inst_key, _ in meta:
        tick_labels.append(f"{TASK_SHORT[task_name]}\n{'PT' if is_pt else 'FT'}\n{inst_key}")
        tick_colors.append(task_point_color(task_name, is_pt))

    # Separators between tasks (thick) and between PT/FT within a task (thin)
    task_sep, model_sep = [], []
    prev_task, prev_is_pt = None, None
    for i, (task_name, is_pt, _, _) in enumerate(meta):
        if i == 0:
            prev_task, prev_is_pt = task_name, is_pt
            continue
        if task_name != prev_task:
            task_sep.append(i)
        elif is_pt != prev_is_pt:
            model_sep.append(i)
        prev_task, prev_is_pt = task_name, is_pt

    cell_size = 0.55
    fig, ax = plt.subplots(figsize=(n * cell_size + 3, n * cell_size + 2))
    im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="cosine similarity", fraction=0.03, pad=0.02)

    # Cell value annotations (small font for large matrix)
    for r in range(n):
        for c in range(n):
            val = sim[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if 0.35 < val < 0.85 else "white")

    # Separator lines
    for sep in model_sep:
        ax.axhline(sep - 0.5, color="white", linewidth=1)
        ax.axvline(sep - 0.5, color="white", linewidth=1)
    for sep in task_sep:
        ax.axhline(sep - 0.5, color="white", linewidth=3)
        ax.axvline(sep - 0.5, color="white", linewidth=3)

    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=5, rotation=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=5)

    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(color)

    ax.set_title(f"all tasks  ·  {emb_type}  cosine similarity", fontsize=11)
    fig.tight_layout()
    _save(fig, output_dir, f"all_tasks_{emb_type}_cosim.png", task_name="all_tasks")


# ── Util ──────────────────────────────────────────────────────────────────────

def _save(fig, output_dir: str, filename: str, task_name: str = None) -> None:
    dest = os.path.join(output_dir, task_name) if task_name else output_dir
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    parser = argparse.ArgumentParser(
        description=(
            "Visualize Pi0 VLM embeddings (T-SNE, PCA, cosine sim). "
            "Pass a task name for per-task plots, or omit for all-tasks combined plots."
        )
    )
    parser.add_argument(
        "task_name", nargs="?", default=None,
        help="e.g. put_carrot_on_plate  (omit to visualize all tasks together)",
    )
    parser.add_argument("--embeddings_dir", default=os.path.join(repo, "embeddings", "pi0"))
    parser.add_argument("--output_dir",     default=os.path.join(repo, "visualization", "pi0"))
    args = parser.parse_args()

    for emb_type in ("all_layers", "last_layer"):
        print(f"\n[{emb_type}]")
        if args.task_name:
            print(f"Task: {args.task_name}")
            make_tsne_plot (args.embeddings_dir, args.task_name, args.output_dir, emb_type)
            make_pca_plot  (args.embeddings_dir, args.task_name, args.output_dir, emb_type)
            make_cosim_plot(args.embeddings_dir, args.task_name, args.output_dir, emb_type)
        else:
            print("All tasks combined")
            make_all_tasks_tsne_plot (args.embeddings_dir, args.output_dir, emb_type)
            make_all_tasks_pca_plot  (args.embeddings_dir, args.output_dir, emb_type)
            make_all_tasks_cosim_plot(args.embeddings_dir, args.output_dir, emb_type)


if __name__ == "__main__":
    main()
