"""
Visualize VLM embeddings (Pi0 or OpenVLA) for a given task or all tasks together.

Per-task mode (positional task_name argument):
  Produces three plots per embedding type:
    - T-SNE scatter
    - PCA scatter  (PC axes show % variance explained)
    - Cosine similarity heatmap

All-tasks mode (no positional argument):
  Same three plot types, combining all 4 tasks in one figure.
  Color encodes task identity; marker/shade encodes weight set.

Use --model to select defaults for each model:
  pi0     : emb_types="all_layers last_layer"   dirs=embeddings/pi0  visualization/pi0
            weight sets: base (PaliGemma) · pre-trained (Pi0 bridge) · co-trained (Pi0 task)
  openvla : emb_types="layer-1_mean layer-1_final"  dirs=embeddings/openvla visualization/openvla
            weight sets: base (Prismatic) · pre-trained (Open-X) · co-trained (finetuned)

Color scheme (scatter plots):
  co-trained  base instruction  → vivid green,  large marker
  co-trained  other variants    → vivid blue,   large marker
  pre-trained (any)             → medium-faded colors
  base        (any)             → most-faded colors
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

# Three fade levels: base (most faded) · pre-trained (medium) · co-trained (vivid)
WEIGHT_SET_ORDER = ["base", "pre-trained", "co-trained"]

WEIGHT_SET_COLORS = {
    "base":        (_fade(C_FT_BASE, 0.15), _fade(C_FT_OTHER, 0.15)),
    "pre-trained": (_fade(C_FT_BASE, 0.55), _fade(C_FT_OTHER, 0.55)),
    "co-trained":  (C_FT_BASE,              C_FT_OTHER),
}

WEIGHT_SET_ALPHA = {"base": 0.15, "pre-trained": 0.50, "co-trained": 1.0}
WEIGHT_SET_SIZE  = {"base": 90,   "pre-trained": 110,  "co-trained": 150}
WEIGHT_SET_ZORD  = {"base": 1,    "pre-trained": 2,    "co-trained": 3}
WS_ABBREV        = {"base": "base", "pre-trained": "PT", "co-trained": "CT"}


def _ws_legend_label(ws: str, ws_present: set) -> str:
    """Return the display label for ws in a legend, annotating fading level.

    With 2 conditions (1 faded set):   "base (faded)"
    With 3 conditions (2 faded sets):  "base (most faded)", "pre-trained (faded)"
    "co-trained" is never annotated.
    """
    if ws == "co-trained":
        return "co-trained"
    faded = [w for w in WEIGHT_SET_ORDER if w != "co-trained" and w in ws_present]
    if len(faded) >= 2 and ws == "base":
        return "base (most faded)"
    return f"{ws} (faded)"


def _ws_color(weight_set: str, is_base_inst: bool) -> str:
    c_base, c_other = WEIGHT_SET_COLORS.get(weight_set, WEIGHT_SET_COLORS["pre-trained"])
    return c_base if is_base_inst else c_other


def task_point_color(task_name: str, weight_set: str) -> str:
    base_color = TASK_COLORS_FT[task_name]
    alpha = WEIGHT_SET_ALPHA.get(weight_set, 0.45)
    return _fade(base_color, alpha=alpha) if alpha < 1.0 else base_color


INST_ORDER = ["base", "S-PROP", "S-LANG", "S-MO", "S-INT"]

def _inst_display(inst_key: str) -> str:
    """Return the display label for an instruction key."""
    return "original" if inst_key == "base" else inst_key

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

ALL_SUFFIXES = ("_basevlm_embds", "_pretrained_embds", "_pi0pretrained_embds", "_finetuned_embds")


def find_embedding_files(embeddings_dir: str, task_name: str, file_suffixes=None):
    """Return sorted .npz paths for task_name, optionally filtered by suffix list."""
    task_dir = os.path.join(embeddings_dir, task_name)
    all_files = sorted(glob.glob(os.path.join(task_dir, f"{task_name}_*.npz")))
    if file_suffixes is None:
        return all_files
    return [f for f in all_files if any(f.endswith(s + ".npz") for s in file_suffixes)]


def parse_file_info(filepath: str, task_name: str, suffix_to_label: dict):
    """Return (weight_set_label: str, model_label: str).

    suffix_to_label maps file suffixes to display labels, e.g.:
      {"_pretrained_embds": "base", "_finetuned_embds": "co-trained"}   (pi0)
      {"_basevlm_embds": "base", "_pretrained_embds": "pre-trained",
       "_finetuned_embds": "co-trained"}                                 (openvla)
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    body = stem[len(task_name) + 1:]
    for suffix, label in suffix_to_label.items():
        if body.endswith(suffix):
            return label, body[: -len(suffix)]
    return None, body


def load_vecs(npz_path: str, emb_type: str) -> dict:
    """Load per-instruction embeddings of a given type from an .npz file.

    Handles both Pi0 shape [1, D] (batch dim present) and OpenVLA shape [D].
    """
    data = np.load(npz_path)
    result = {}
    for key in INST_ORDER:
        full_key = f"{key}_{emb_type}"
        if full_key in data:
            vec = data[full_key]
            result[key] = vec[0] if vec.ndim == 2 else vec
    return result


def collect_embeddings(embeddings_dir: str, task_name: str, emb_type: str,
                       suffix_to_label: dict, file_suffixes=None):
    """
    Returns
    -------
    X    : np.ndarray [n, dim]
    meta : list of (weight_set: str, inst_key: str, model_label: str)
    """
    files = find_embedding_files(embeddings_dir, task_name, file_suffixes)
    vecs, meta = [], []
    for fp in files:
        ws, model_label = parse_file_info(fp, task_name, suffix_to_label)
        if ws is None:
            continue
        for inst_key, vec in load_vecs(fp, emb_type).items():
            vecs.append(vec)
            meta.append((ws, inst_key, model_label))
    if not vecs:
        return None, None
    return np.stack(vecs), meta


def collect_all_tasks_embeddings(embeddings_dir: str, emb_type: str,
                                  suffix_to_label: dict, file_suffixes=None):
    """
    Returns
    -------
    X    : np.ndarray [n, dim]
    meta : list of (task_name: str, weight_set: str, inst_key: str, model_label: str)

    file_suffixes filters which weight sets to include (None = all recognised suffixes).
    Ordering within each task: base → pre-trained → co-trained.
    """
    suffixes = file_suffixes if file_suffixes is not None else ALL_SUFFIXES
    suffix_order = [s for s in ALL_SUFFIXES if s in suffixes]

    vecs, meta = [], []
    for task_name in TASK_ORDER:
        all_files = find_embedding_files(embeddings_dir, task_name)
        if not all_files:
            print(f"  [warn] no embedding files for task: {task_name}")
            continue
        ordered = []
        for sfx in suffix_order:
            ordered += [f for f in all_files if f.endswith(sfx + ".npz")]
        for fp in ordered:
            ws, model_label = parse_file_info(fp, task_name, suffix_to_label)
            if ws is None:
                continue
            for inst_key, vec in load_vecs(fp, emb_type).items():
                vecs.append(vec)
                meta.append((task_name, ws, inst_key, model_label))
    if not vecs:
        return None, None
    return np.stack(vecs), meta


# ── Single-task scatter renderer ──────────────────────────────────────────────

def _render_scatter(ax, X2d, meta, xlabel: str, ylabel: str, title: str) -> None:
    used_ws = set()
    for i, (ws, inst_key, _) in enumerate(meta):
        is_base = inst_key == "base"
        color = _ws_color(ws, is_base)
        ax.scatter(
            X2d[i, 0], X2d[i, 1],
            c=color, s=WEIGHT_SET_SIZE.get(ws, 110),
            marker="o", edgecolors="#555", linewidths=0.6,
            zorder=WEIGHT_SET_ZORD.get(ws, 2),
        )
        ax.annotate(
            _inst_display(inst_key), (X2d[i, 0], X2d[i, 1]),
            textcoords="offset points", xytext=(6, 5),
            fontsize=8, color=color,
            fontweight="bold" if ws == "co-trained" else "normal",
        )
        used_ws.add(ws)

    handles = []
    for ws in WEIGHT_SET_ORDER:
        if ws not in used_ws:
            continue
        lbl = _ws_legend_label(ws, used_ws)
        c_base, c_other = WEIGHT_SET_COLORS[ws]
        handles.append(mpatches.Patch(facecolor=c_base,  edgecolor="#555",
                                       linewidth=0.6, label=f"{lbl} — original"))
        handles.append(mpatches.Patch(facecolor=c_other, edgecolor="#555",
                                       linewidth=0.6, label=f"{lbl} — variant"))
    ax.legend(handles=handles, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")


# ── All-tasks scatter renderer ────────────────────────────────────────────────

def _render_all_tasks_scatter(ax, X2d, meta, xlabel: str, ylabel: str, title: str) -> None:
    for i, (task_name, ws, inst_key, _) in enumerate(meta):
        color = task_point_color(task_name, ws)
        marker = "D" if inst_key == "base" else "o"
        ax.scatter(
            X2d[i, 0], X2d[i, 1],
            c=color, s=WEIGHT_SET_SIZE.get(ws, 110), marker=marker,
            edgecolors="#555", linewidths=0.5,
            zorder=WEIGHT_SET_ZORD.get(ws, 2),
        )
        ax.annotate(
            f"{TASK_SHORT[task_name]}\n{_inst_display(inst_key)}",
            (X2d[i, 0], X2d[i, 1]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=6, color=color,
            fontweight="bold" if ws == "co-trained" else "normal",
        )

    # Legend: task colors
    task_handles = [
        mpatches.Patch(
            facecolor=TASK_COLORS_FT[t], edgecolor="#555", linewidth=0.6,
            label=TASK_SHORT[t],
        )
        for t in TASK_ORDER if any(m[0] == t for m in meta)
    ]

    # Legend: weight-set shade/marker encoding (only sets present in data)
    ws_present = sorted(set(m[1] for m in meta),
                        key=lambda w: WEIGHT_SET_ORDER.index(w) if w in WEIGHT_SET_ORDER else 99)
    shade_map = {"co-trained": "#666", "pre-trained": "#aaa", "base": "#ccc"}
    enc_handles = []
    for ws in ws_present:
        lbl = _ws_legend_label(ws, set(ws_present))
        shade = shade_map.get(ws, "#aaa")
        sz = WEIGHT_SET_SIZE.get(ws, 110)
        enc_handles.append(plt.scatter([], [], marker="D", c=shade, s=sz,
                                        edgecolors="#555", linewidths=0.5,
                                        label=f"{lbl} — original"))
        enc_handles.append(plt.scatter([], [], marker="o", c=shade, s=sz,
                                        edgecolors="#555", linewidths=0.5,
                                        label=f"{lbl} — variant"))

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

def make_tsne_plot(embeddings_dir, task_name, output_dir, emb_type,
                   suffix_to_label, file_suffixes=None) -> None:
    X, meta = collect_embeddings(embeddings_dir, task_name, emb_type,
                                  suffix_to_label, file_suffixes)
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


def make_all_tasks_tsne_plot(embeddings_dir, output_dir, emb_type,
                             suffix_to_label, file_suffixes=None) -> None:
    X, meta = collect_all_tasks_embeddings(embeddings_dir, emb_type,
                                            suffix_to_label, file_suffixes)
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

def make_pca_plot(embeddings_dir, task_name, output_dir, emb_type,
                  suffix_to_label, file_suffixes=None) -> None:
    X, meta = collect_embeddings(embeddings_dir, task_name, emb_type,
                                  suffix_to_label, file_suffixes)
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


def make_all_tasks_pca_plot(embeddings_dir, output_dir, emb_type,
                            suffix_to_label, file_suffixes=None) -> None:
    X, meta = collect_all_tasks_embeddings(embeddings_dir, emb_type,
                                            suffix_to_label, file_suffixes)
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

def make_cosim_plot(embeddings_dir, task_name, output_dir, emb_type,
                    suffix_to_label, file_suffixes=None) -> None:
    X, meta = collect_embeddings(embeddings_dir, task_name, emb_type,
                                  suffix_to_label, file_suffixes)
    if X is None or len(X) < 2:
        print(f"  [CosSim] not enough data"); return

    sim = cosine_similarity(X)
    n = len(meta)

    # Group by (weight_set, model_label); order: base → pre-trained → co-trained
    seen = []
    for ws, _, ml in meta:
        if (ws, ml) not in seen:
            seen.append((ws, ml))
    seen.sort(key=lambda x: (
        WEIGHT_SET_ORDER.index(x[0]) if x[0] in WEIGHT_SET_ORDER else 99,
        x[1],
    ))

    ordered_idx = []
    for (ws, ml) in seen:
        for inst_key in INST_ORDER:
            for i, (iws, ik, iml) in enumerate(meta):
                if iws == ws and ik == inst_key and iml == ml:
                    ordered_idx.append(i)

    sim_ord = sim[np.ix_(ordered_idx, ordered_idx)]
    meta_ord = [meta[i] for i in ordered_idx]

    multi_model = len(seen) > 1

    def short_label(ws, inst_key, _):
        if not multi_model:
            return _inst_display(inst_key)
        return f"{WS_ABBREV.get(ws, ws)}\n{_inst_display(inst_key)}"

    tick_labels = [short_label(*m) for m in meta_ord]
    tick_colors = [_ws_color(m[0], m[1] == "base") for m in meta_ord]

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


def make_all_tasks_cosim_plot(embeddings_dir, output_dir, emb_type,
                              suffix_to_label, file_suffixes=None) -> None:
    X, meta = collect_all_tasks_embeddings(embeddings_dir, emb_type,
                                            suffix_to_label, file_suffixes)
    if X is None or len(X) < 2:
        print(f"  [CosSim all-tasks] not enough data"); return

    n = len(meta)
    sim = cosine_similarity(X)

    tick_labels, tick_colors = [], []
    for task_name, ws, inst_key, _ in meta:
        tick_labels.append(
            f"{TASK_SHORT[task_name]}\n{WS_ABBREV.get(ws, ws)}\n{_inst_display(inst_key)}"
        )
        tick_colors.append(task_point_color(task_name, ws))

    # Separators: thick between tasks, thin between weight sets within a task
    task_sep, model_sep = [], []
    prev_task, prev_ws = None, None
    for i, (task_name, ws, _, _) in enumerate(meta):
        if i == 0:
            prev_task, prev_ws = task_name, ws
            continue
        if task_name != prev_task:
            task_sep.append(i)
        elif ws != prev_ws:
            model_sep.append(i)
        prev_task, prev_ws = task_name, ws

    cell_size = 0.55
    fig, ax = plt.subplots(figsize=(n * cell_size + 3, n * cell_size + 2))
    im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="cosine similarity", fraction=0.03, pad=0.02)

    for r in range(n):
        for c in range(n):
            val = sim[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if 0.35 < val < 0.85 else "white")

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

MODEL_DEFAULTS = {
    "pi0": {
        "emb_types":      ["all_layers", "last_layer"],
        "subdir":         "pi0",
        "suffix_to_label": {
            "_pretrained_embds":    "base",        # PaliGemma base model
            "_pi0pretrained_embds": "pre-trained", # Pi0 bridge-pretrained
            "_finetuned_embds":     "co-trained",  # Pi0 task-specific fine-tuned
        },
    },
    "openvla": {
        "emb_types":      ["layer-1_mean", "layer-1_final"],
        "subdir":         "openvla",
        "suffix_to_label": {
            "_basevlm_embds":    "base",        # Prismatic VLM (pre-robot)
            "_pretrained_embds": "pre-trained", # OpenVLA trained on Open-X
            "_finetuned_embds":  "co-trained",  # fine-tuned checkpoint
        },
    },
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    parser = argparse.ArgumentParser(
        description=(
            "Visualize VLM embeddings (T-SNE, PCA, cosine sim). "
            "Pass a task name for per-task plots, or omit for all-tasks combined plots."
        )
    )
    parser.add_argument(
        "task_name", nargs="?", default=None,
        help="e.g. put_carrot_on_plate  (omit to visualize all tasks together)",
    )
    parser.add_argument(
        "--model", choices=list(MODEL_DEFAULTS), default="pi0",
        help="Which model's embeddings to visualize. Sets defaults for --embeddings_dir, "
             "--output_dir, and --emb_types. (default: pi0)",
    )
    parser.add_argument("--embeddings_dir", default=None,
                        help="Override the default embeddings directory for --model.")
    parser.add_argument("--output_dir",     default=None,
                        help="Override the default visualization output directory for --model.")
    parser.add_argument(
        "--emb_types", nargs="+", default=None,
        help="Override embedding types to plot (default depends on --model).",
    )
    parser.add_argument(
        "--file_suffixes", nargs="+", default=None,
        metavar="SUFFIX",
        help="Restrict which weight sets to include, by file suffix "
             "(e.g. _pretrained_embds _finetuned_embds). Default: all recognised suffixes.",
    )
    args = parser.parse_args()

    defaults = MODEL_DEFAULTS[args.model]
    if args.embeddings_dir is None:
        args.embeddings_dir = os.path.join(repo, "embeddings", defaults["subdir"])
    if args.output_dir is None:
        args.output_dir = os.path.join(repo, "visualization", defaults["subdir"])
    emb_types      = args.emb_types if args.emb_types is not None else defaults["emb_types"]
    suffix_to_label = defaults["suffix_to_label"]
    file_suffixes   = args.file_suffixes  # None → all

    for emb_type in emb_types:
        print(f"\n[{emb_type}]")
        if args.task_name:
            print(f"Task: {args.task_name}")
            make_tsne_plot (args.embeddings_dir, args.task_name, args.output_dir,
                            emb_type, suffix_to_label, file_suffixes)
            make_pca_plot  (args.embeddings_dir, args.task_name, args.output_dir,
                            emb_type, suffix_to_label, file_suffixes)
            make_cosim_plot(args.embeddings_dir, args.task_name, args.output_dir,
                            emb_type, suffix_to_label, file_suffixes)
        else:
            print("All tasks combined")
            make_all_tasks_tsne_plot (args.embeddings_dir, args.output_dir,
                                      emb_type, suffix_to_label, file_suffixes)
            make_all_tasks_pca_plot  (args.embeddings_dir, args.output_dir,
                                      emb_type, suffix_to_label, file_suffixes)
            make_all_tasks_cosim_plot(args.embeddings_dir, args.output_dir,
                                      emb_type, suffix_to_label, file_suffixes)


if __name__ == "__main__":
    main()
