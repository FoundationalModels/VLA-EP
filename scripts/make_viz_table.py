#!/usr/bin/env python3
"""Build summary tables and save them to visualizations/.

Table 1 — summary_table.png  (original)
  Rows: comparison subsets (base+pretrained, pretrained+cotrained, …)
  Cols: models (Pi0, OpenVLA, MiniVLA)
  Cell: all-tasks joint PCA for that subset

Table 2 — summary_pairwise_pca.png
  Single row: pairwise PCA from base+pretrained+cotrained subset
  Cols: models (Pi0, OpenVLA, MiniVLA)
  Each cell is a 3-panel figure (base vs pre-trained | pre-trained vs co-trained
  | base vs co-trained) so one row conveys all three pairwise comparisons.

Table 3 — summary_pairwise_pca_by_checkpoint.png
  Rows: models (Pi0, OpenVLA, MiniVLA)
  Single image column: pairwise PCA panels, independently fitted for the
  carrot/knife and pot/plate co-trained checkpoint task groups.

Table 4 — summary_discrimination_sensitivity_lines.png
  Line plot of between-task discrimination and within-task instruction
  sensitivity across base, pre-trained, and co-trained weight stages.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from make_discrimination_sensitivity_summary import main as make_line_summary

VIZ_ROOT = Path("/home/hrilab/VLA-EP/visualizations")

MODELS       = ["pi0",  "openvla",                    "minivla"]
MODEL_LABELS = ["Pi0",  "OpenVLA",                    "MiniVLA"]

# Per-model filename stems for the two plot types
JOINT_PCA_PATTERNS = {
    "pi0":     "all_tasks_last_layer_pca.png",
    "openvla": "all_tasks_layer-1_final_pca.png",
    "minivla": "all_tasks_layer-1_final_pca.png",
}
PAIRWISE_PCA_PATTERNS = {
    "pi0":     "all_tasks_last_layer_pairwise_pca.png",
    "openvla": "all_tasks_layer-1_final_pairwise_pca.png",
    "minivla": "all_tasks_layer-1_final_pairwise_pca.png",
}
CHECKPOINT_GROUP_PCA_PATTERNS = {
    "pi0":     "all_tasks_last_layer_checkpoint_group_pairwise_pca.png",
    "openvla": "all_tasks_layer-1_final_checkpoint_group_pairwise_pca.png",
    "minivla": "all_tasks_layer-1_final_checkpoint_group_pairwise_pca.png",
}

COMPARISONS = [
    ("base+pretrained",          "Base vs\nPre-trained"),
    ("pretrained+cotrained",     "Pre-trained vs\nCo-trained"),
    ("base+cotrained",           "Base vs\nCo-trained"),
    ("base+pretrained+cotrained","All Three"),
]

try:
    _font_hdr = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    _font_lbl = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except OSError:
    _font_hdr = ImageFont.load_default()
    _font_lbl = _font_hdr

BG_COLOR   = (255, 255, 255)
HEADER_BG  = (220, 225, 235)
LABEL_BG   = (240, 242, 246)
TEXT_COLOR = (25,  30,  40)
MISSING_BG = (200, 200, 200)
MISSING_FG = (180,   0,   0)
PADDING    = 10


def _centered_text(draw, text, font, rect):
    x0, y0, x1, y1 = rect
    lines = text.split("\n")
    line_bboxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    line_hs = [b[3] - b[1] for b in line_bboxes]
    total_h = sum(line_hs) + 4 * (len(lines) - 1)
    cy = y0 + (y1 - y0 - total_h) // 2
    for ln, bbox, lh in zip(lines, line_bboxes, line_hs):
        tw = bbox[2] - bbox[0]
        draw.text((x0 + (x1 - x0 - tw) // 2, cy), ln, fill=TEXT_COLOR, font=font)
        cy += lh + 4


def _paste_or_placeholder(canvas, draw, img_path, x, y, w, h):
    if img_path.exists():
        img = Image.open(img_path).convert("RGB").resize((w, h), Image.LANCZOS)
        canvas.paste(img, (x, y))
    else:
        draw.rectangle([x, y, x + w, y + h], fill=MISSING_BG)
        draw.text((x + 12, y + h // 2 - 10),
                  f"Missing:\n{img_path.name}", fill=MISSING_FG, font=_font_lbl)
        print(f"WARNING: {img_path} not found")


# ═══════════════════════════════════════════════════════════════════════════════
# Table 1 — original: comparison subsets × models, joint PCA
# ═══════════════════════════════════════════════════════════════════════════════

CELL_W1      = 640
CELL_H1      = 522
COL_HEADER_H = 64
ROW_LABEL_W  = 220

n_rows1 = len(COMPARISONS)
n_cols  = len(MODELS)
total_w1 = ROW_LABEL_W + n_cols * (CELL_W1 + PADDING) + PADDING
total_h1 = COL_HEADER_H + n_rows1 * (CELL_H1 + PADDING) + PADDING

canvas1 = Image.new("RGB", (total_w1, total_h1), BG_COLOR)
draw1   = ImageDraw.Draw(canvas1)

for col_idx, label in enumerate(MODEL_LABELS):
    x = ROW_LABEL_W + col_idx * (CELL_W1 + PADDING) + PADDING
    draw1.rectangle([x, 0, x + CELL_W1, COL_HEADER_H], fill=HEADER_BG)
    _centered_text(draw1, label, _font_hdr, (x, 0, x + CELL_W1, COL_HEADER_H))

for row_idx, (subset, row_label) in enumerate(COMPARISONS):
    y = COL_HEADER_H + row_idx * (CELL_H1 + PADDING) + PADDING
    draw1.rectangle([0, y, ROW_LABEL_W, y + CELL_H1], fill=LABEL_BG)
    _centered_text(draw1, row_label, _font_lbl, (0, y, ROW_LABEL_W, y + CELL_H1))
    for col_idx, model in enumerate(MODELS):
        x = ROW_LABEL_W + col_idx * (CELL_W1 + PADDING) + PADDING
        img_path = VIZ_ROOT / model / subset / "all_tasks" / JOINT_PCA_PATTERNS[model]
        _paste_or_placeholder(canvas1, draw1, img_path, x, y, CELL_W1, CELL_H1)

out1 = VIZ_ROOT / "summary_table.png"
canvas1.save(out1, dpi=(150, 150))
print(f"Saved → {out1}  ({total_w1} × {total_h1} px)")


# ═══════════════════════════════════════════════════════════════════════════════
# Table 2 — pairwise PCA: models as rows, single image column (vertical stack)
# ═══════════════════════════════════════════════════════════════════════════════

CELL_W2 = 1800   # wide so the 3-panel figure stays legible
CELL_H2 = 508    # preserves ~3.5:1 aspect ratio of the pairwise PCA figure

total_w2 = ROW_LABEL_W + CELL_W2 + 2 * PADDING
total_h2 = COL_HEADER_H + len(MODELS) * (CELL_H2 + PADDING) + PADDING

canvas2 = Image.new("RGB", (total_w2, total_h2), BG_COLOR)
draw2   = ImageDraw.Draw(canvas2)

# Single column header
draw2.rectangle([ROW_LABEL_W + PADDING, 0, total_w2, COL_HEADER_H], fill=HEADER_BG)
_centered_text(draw2, "Pairwise PCA  (base · pre-trained · co-trained)",
               _font_hdr, (ROW_LABEL_W + PADDING, 0, total_w2, COL_HEADER_H))

# Top-left corner cell
draw2.rectangle([0, 0, ROW_LABEL_W, COL_HEADER_H], fill=HEADER_BG)

for row_idx, (model, label) in enumerate(zip(MODELS, MODEL_LABELS)):
    y = COL_HEADER_H + row_idx * (CELL_H2 + PADDING) + PADDING
    # Row label
    draw2.rectangle([0, y, ROW_LABEL_W, y + CELL_H2], fill=LABEL_BG)
    _centered_text(draw2, label, _font_hdr, (0, y, ROW_LABEL_W, y + CELL_H2))
    # Image
    x = ROW_LABEL_W + PADDING
    img_path = (VIZ_ROOT / model / "base+pretrained+cotrained"
                / "all_tasks" / PAIRWISE_PCA_PATTERNS[model])
    _paste_or_placeholder(canvas2, draw2, img_path, x, y, CELL_W2, CELL_H2)

out2 = VIZ_ROOT / "summary_pairwise_pca.png"
canvas2.save(out2, dpi=(150, 150))
print(f"Saved → {out2}  ({total_w2} × {total_h2} px)")


# ═══════════════════════════════════════════════════════════════════════════════
# Table 3 — checkpoint-group pairwise PCA: models as rows
# ═══════════════════════════════════════════════════════════════════════════════

CELL_W3 = 1800
CELL_H3 = 950

total_w3 = ROW_LABEL_W + CELL_W3 + 2 * PADDING
total_h3 = COL_HEADER_H + len(MODELS) * (CELL_H3 + PADDING) + PADDING

canvas3 = Image.new("RGB", (total_w3, total_h3), BG_COLOR)
draw3   = ImageDraw.Draw(canvas3)

draw3.rectangle([ROW_LABEL_W + PADDING, 0, total_w3, COL_HEADER_H], fill=HEADER_BG)
_centered_text(draw3, "Pairwise PCA by checkpoint  (carrot/knife · pot/plate)",
               _font_hdr, (ROW_LABEL_W + PADDING, 0, total_w3, COL_HEADER_H))
draw3.rectangle([0, 0, ROW_LABEL_W, COL_HEADER_H], fill=HEADER_BG)

for row_idx, (model, label) in enumerate(zip(MODELS, MODEL_LABELS)):
    y = COL_HEADER_H + row_idx * (CELL_H3 + PADDING) + PADDING
    draw3.rectangle([0, y, ROW_LABEL_W, y + CELL_H3], fill=LABEL_BG)
    _centered_text(draw3, label, _font_hdr, (0, y, ROW_LABEL_W, y + CELL_H3))

    x = ROW_LABEL_W + PADDING
    img_path = (VIZ_ROOT / model / "base+pretrained+cotrained"
                / "all_tasks" / CHECKPOINT_GROUP_PCA_PATTERNS[model])
    _paste_or_placeholder(canvas3, draw3, img_path, x, y, CELL_W3, CELL_H3)

out3 = VIZ_ROOT / "summary_pairwise_pca_by_checkpoint.png"
canvas3.save(out3, dpi=(150, 150))
print(f"Saved → {out3}  ({total_w3} × {total_h3} px)")


# ═══════════════════════════════════════════════════════════════════════════════
# Table 4 — discrimination/sensitivity line summary
# ═══════════════════════════════════════════════════════════════════════════════

make_line_summary()
