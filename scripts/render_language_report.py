#!/usr/bin/env python3
"""Render captured VLA language output as a readable Markdown report.

get_vla_language_output.py writes one JSON per checkpoint, which is fine for
downstream code but awkward to read. This flattens them into a single document
grouped by task and instruction variant, with the reasoning text inline and the
instruction it came from alongside it.

Usage:
  python scripts/render_language_report.py
  python scripts/render_language_report.py --output /tmp/report.md
"""

import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

INST_ORDER = ["base", "S-PROP", "S-LANG", "S-MO", "S-INT"]

# Nicer ordering than glob's: reasoner first, since it is the only one that
# produces language, then its matched control, then the plain baseline.
CHECKPOINT_ORDER = ["reasoner", "steerable", "pretrained"]


def load_ref_exp(path):
    with open(path) as f:
        return json.loads(re.sub(r"//[^\n]*", "", f.read()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language_dir", default=os.path.join(REPO, "visualizations", "language"))
    p.add_argument("--ref_exp_path", default=os.path.join(REPO, "media", "ref_exp.jsonc"))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    ref = load_ref_exp(args.ref_exp_path)

    files = {}
    for fp in glob.glob(os.path.join(args.language_dir, "*_language.json")):
        name = os.path.basename(fp)[: -len("_language.json")]
        with open(fp) as f:
            files[name] = json.load(f)
    if not files:
        raise SystemExit(f"No *_language.json found in {args.language_dir}")

    names = ([n for n in CHECKPOINT_ORDER if n in files]
             + sorted(n for n in files if n not in CHECKPOINT_ORDER))

    out = ["# VLA language output", ""]

    # ── Summary ──
    out += ["## Summary", "",
            "| checkpoint | prime | variants producing reasoning | tokens (min–max) |",
            "|---|---|---|---|"]
    for n in names:
        d = files[n]
        recs = [r for t in d["results"].values() for r in t.values()]
        # A couple of tokens is punctuation/EOS, not language.
        talk = [r for r in recs if r["n_text_tokens"] > 5]
        rng = (f"{min(r['n_text_tokens'] for r in talk)}–"
               f"{max(r['n_text_tokens'] for r in talk)}") if talk else "—"
        out.append(f"| `{n}` | `{d.get('prime','')!r}` | **{len(talk)} / {len(recs)}** | {rng} |")
    out.append("")

    # ── Per-checkpoint detail ──
    for n in names:
        d = files[n]
        out += [f"## `{n}`", "",
                f"- checkpoint: `{d['checkpoint']}`",
                f"- prime: `{d.get('prime','')!r}`   max_new_tokens: {d.get('max_new_tokens')}",
                ""]

        for task, variants in d["results"].items():
            out += [f"### {task}", ""]
            instructions = {"base": ref[task]["base_command"]}
            instructions.update(ref[task]["variants"])

            for key in [k for k in INST_ORDER if k in variants]:
                r = variants[key]
                out += [f"**{key}** — _{instructions.get(key, '?')}_  "
                        f"<sub>{r['n_text_tokens']} text / {r['n_action_tokens']} action tokens</sub>",
                        ""]
                text = r["text"].strip()
                if r["n_text_tokens"] > 5 and text:
                    out += ["```text", text, "```", ""]
                else:
                    out += ["> _(no language emitted)_", ""]

    md = "\n".join(out)
    dest = args.output or os.path.join(args.language_dir, "language_report.md")
    with open(dest, "w") as f:
        f.write(md)
    print(f"Saved → {dest}  ({len(md)} chars)")


if __name__ == "__main__":
    main()
