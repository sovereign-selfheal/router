#!/usr/bin/env python3
"""Charts of docs/c1-off-eval-2026-10-05.md, from the raw data in perf/results/.

    uv run --with matplotlib==3.10.7 perf/plot_c1_off.py

Writes SVG files in docs/img/. Inputs (prefix 2026-10-05-ocp.rw287):
- the per-case CSV files of eval/run_eval.py: <prefix>-eval-<config>-<set>.csv, config `on` / `off`
  (C1 on or off, threshold 0.50), `onr` / `offr` (threshold 0.70) and `chain-on` / `chain-off`;
- the JSON lines of the gate scenario of perf/c2_perf.py: <prefix>-gate.jsonl;
- the agent contexts K1-K10 of the two validation runs: <prefix>-k-first-chunk.json.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "perf/results"
OUT = ROOT / "docs/img"
PREFIX = "2026-10-05-ocp.rw287"

# The palette of plot_c2_questions.py; C1 on and off keep the colors of Q1 and Q2 there.
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8e5bd1"
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODES = (("C1 on (Presidio NER)", "on", ORANGE), ("C1 off", "off", AQUA))

# (label on the axis, configuration suffix, set): the threshold is in the suffix.
SETS = (
    ("english\n0.50", "", "privacy-plus-english"),
    ("italian\n0.50", "", "privacy-plus-italian"),
    ("english\n0.70", "r", "privacy-plus-english"),
    ("italian\n0.70", "r", "privacy-plus-italian"),
    ("agent contexts\n0.70", "r", "agent-contexts"),
    ("chain-demo\nchain mode", "chain", "chain-demo"),
)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def eval_rows(mode, suffix, set_name):
    config = f"chain-{mode}" if suffix == "chain" else f"{mode}{suffix}"
    with (RESULTS / f"{PREFIX}-eval-{config}-{set_name}.csv").open() as fh:
        return list(csv.DictReader(fh))


def errors(rows):
    leaks = sum(r["decision"] != r["expect"] and r["expect"] == "local" for r in rows)
    fps = sum(r["decision"] != r["expect"] and r["expect"] == "sota" for r in rows)
    return leaks, fps


def quality_chart():
    """Leaks and false positives per set and threshold, C1 on and off."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    width = 0.38
    for ax, kind, title in (
        (axes[0], 0, "Leaks (sensitive text sent to SOTA)"),
        (axes[1], 1, "False positives (benign text kept LOCAL)"),
    ):
        style(ax, title, "cases")
        top = 0
        for i, (_label, mode, color) in enumerate(MODES):
            for j, (_name, suffix, set_name) in enumerate(SETS):
                value = errors(eval_rows(mode, suffix, set_name))[kind]
                top = max(top, value)
                x = j + (i - 0.5) * width
                ax.bar(x, value, width - 0.04, color=color, edgecolor=SURFACE, linewidth=1.5)
                ax.text(x, value, f"{value}\n", color=TEXT, fontsize=8, ha="center", va="bottom")
        ax.set_ylim(0, max(4, top * 1.25))
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels([f"{name}\n({len(eval_rows('on', suffix, set_name))})"
                            for name, suffix, set_name in SETS], color=MUTED, fontsize=8)
    for label, _mode, color in MODES:
        axes[0].bar(0, 0, color=color, label=label)
    fig.legend(frameon=False, loc="lower center", ncol=2, labelcolor=TEXT, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "c1-off-quality.svg", facecolor=SURFACE)
    plt.close(fig)


def gate_rows():
    path = RESULTS / f"{PREFIX}-gate.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def line(ax, xs, ys, color, label, **kw):
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, label=label, **kw)


def gate_size_chart():
    """Gate time of one agent by context size (a new context every call), C1 on and off."""
    rows = [r for r in gate_rows() if r["concurrency"] == 1]
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    style(ax, "Privacy gate time of one agent, by context size (p50 of 3 new contexts)",
          "seconds")
    for label, mode, color in MODES:
        pts = sorted((r["chars"] / 1000, r["p50"] / 1000, r["ner_error"] + r["c2_error"])
                     for r in rows if r["ner"] == mode)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        line(ax, xs, ys, color, label)
        for x, y, failed in pts:
            # C1 on below its line, C1 off above: the two labels of the smallest size overlap.
            ax.text(x, y, f"  {y:.1f} s" + ("  (timeout: LOCAL)" if failed else ""), color=TEXT,
                    fontsize=8, va="top" if mode == "on" else "bottom")
    ax.axvline(150, color=MUTED, linewidth=1, linestyle=":")
    ax.text(152, ax.get_ylim()[1] * 0.92, "SOTA size cap (150,000 characters):\nabove it the "
            "router keeps the\nrequest LOCAL without the gate", color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("agent context, thousands of characters (about 2.2 characters per token)",
                  color=MUTED)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "c1-off-gate-size.svg", facecolor=SURFACE)
    plt.close(fig)


def gate_parallel_chart():
    """Gate time with 1, 4 and 8 agents at the same time, each with a new context, for two
    context sizes; above each bar, the calls that a detector failure decided (a Presidio or C2
    timeout, or the C2 fallback): these stay LOCAL whatever the text."""
    rows = gate_rows()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE, sharey=True)
    width = 0.38
    for ax, size in zip(axes, (16000, 33000), strict=True):
        sel = [r for r in rows if r["size"] == size]
        chars = round(sum(r["chars"] for r in sel) / len(sel) / 1000)
        style(ax, f"New contexts of about {chars},000 characters", "seconds" if size == 16000
              else "")
        levels = sorted({r["concurrency"] for r in sel})
        for i, (label, mode, color) in enumerate(MODES):
            by = {r["concurrency"]: r for r in sel if r["ner"] == mode}
            xs = [j + (i - 0.5) * width for j in range(len(levels))]
            p50 = [by[c]["p50"] / 1000 for c in levels]
            p95 = [by[c]["p95"] / 1000 for c in levels]
            ax.bar(xs, p95, width - 0.04, color=color, alpha=0.35, edgecolor=SURFACE)
            ax.bar(xs, p50, width - 0.04, color=color, edgecolor=SURFACE,
                   label=label if size == 16000 else None)
            for x, c, top in zip(xs, levels, p95, strict=True):
                r = by[c]
                failed = r["ner_error"] + r["c2_error"] + r["c2_fallback"]
                ax.text(x, top, f"{r['p50'] / 1000:.1f} s\n{failed}/{r['n']} failed\n",
                        color=TEXT, fontsize=7, ha="center", va="bottom")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{c} agent{'s' if c > 1 else ''}" for c in levels], color=MUTED)
    axes[1].set_ylim(0, 40)
    axes[1].text(1.0, 1.02, "dark: p50, light: p95", transform=axes[1].transAxes, ha="right",
                 color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, labelcolor=TEXT,
               fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "c1-off-gate-parallel.svg", facecolor=SURFACE)
    plt.close(fig)


def first_chunk_chart():
    """First chunk of K1-K10 in the two validation runs, with the route of each request."""
    data = json.loads((RESULTS / f"{PREFIX}-k-first-chunk.json").read_text())
    ids = [r["id"] for r in data["on"]]
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=SURFACE)
    style(ax, "Agent contexts K1-K10 through the router: time to the first chunk", "seconds")
    width = 0.38
    for i, (_label, mode, color) in enumerate(MODES):
        by = {r["id"]: r for r in data[mode]}
        for j, k in enumerate(ids):
            r = by[k]
            x = j + (i - 0.5) * width
            sota = r["routed_to"] == "sota-smart"
            ax.bar(x, r["first_chunk_s"], width - 0.04, color=color, edgecolor=TEXT if sota else
                   SURFACE, hatch="//" if sota else None, linewidth=0.8)
            ax.text(x, r["first_chunk_s"], f"{'SOTA' if sota else 'LOCAL'}\n", color=TEXT,
                    fontsize=6.5, ha="center", va="bottom")
    on = {r["id"]: r for r in data["on"]}
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([f"{k}\n{on[k]['prompt_tokens'] // 1000}k tok\n"
                        f"{'sensitive' if 'sensitive' in on[k]['name'] else 'benign'}"
                        for k in ids], color=MUTED, fontsize=8)
    handles = [Patch(facecolor=color, label=label) for label, _mode, color in MODES]
    handles.append(Patch(facecolor=SURFACE, edgecolor=TEXT, hatch="//",
                         label="hatched: routed to SOTA (Gemini)"))
    ax.legend(handles=handles, frameon=False, labelcolor=TEXT, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "c1-off-first-chunk.svg", facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    quality_chart()
    gate_size_chart()
    gate_parallel_chart()
    first_chunk_chart()


if __name__ == "__main__":
    main()
