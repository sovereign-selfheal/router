#!/usr/bin/env python3
"""Charts of docs/c2-questions-eval-2026-10-03.md, from the raw data in perf/results/.

    uv run --with matplotlib==3.10.7 perf/plot_c2_questions.py

Writes SVG files in docs/img/. Inputs: the per-case CSV files of eval/run_eval.py
(2026-10-03-ocp.5bdlz-eval-<config>-<set>.csv) and the JSON lines of perf/c2_perf.py and
perf/agent_growth.py (2026-10-03-ocp.5bdlz-<tag>-<scenario>.jsonl).
"""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "perf/results"
OUT = ROOT / "docs/img"
PREFIX = "2026-10-03-ocp.5bdlz"

# Validated categorical slots 1-4 (blue, orange, aqua, purple) and neutral text colors, as in
# plot_context_cap.py.
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8e5bd1"
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# The three configurations of the evaluation: (label, eval config, agent-contexts config).
CONFIGS = (
    ("Q0: today (A+B+C1, one C2 question)", "q0", "q0r", BLUE),
    ("Q1: split questions, with C1", "q1", "q1r", ORANGE),
    ("Q2: split questions, without C1", "q2", "q2r", AQUA),
)
SETS = (
    ("english", "privacy-plus-english"),
    ("italian", "privacy-plus-italian"),
    ("agent contexts", "agent-contexts"),
    ("chain-demo", "chain-demo"),
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


def eval_rows(config, set_name):
    # chain-demo runs in chain mode (efficiency gate first): its files are chain-<config>-...
    if set_name == "chain-demo":
        config = "chain-" + config.rstrip("r")
    path = RESULTS / f"{PREFIX}-eval-{config}-{set_name}.csv"
    with path.open() as fh:
        return list(csv.DictReader(fh))


def errors(rows):
    leaks = sum(r["decision"] != r["expect"] and r["expect"] == "local" for r in rows)
    fps = sum(r["decision"] != r["expect"] and r["expect"] == "sota" for r in rows)
    return leaks, fps


def quality_chart():
    """Leaks and false positives per configuration and set."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    width = 0.26
    for ax, kind, title in (
        (axes[0], 0, "Leaks (sensitive text sent to SOTA)"),
        (axes[1], 1, "False positives (benign text kept LOCAL)"),
    ):
        style(ax, title, "cases")
        for i, (_label, cfg, cfg_r, color) in enumerate(CONFIGS):
            for j, (_name, set_name) in enumerate(SETS):
                rows = eval_rows(cfg_r if set_name == "agent-contexts" else cfg, set_name)
                value = errors(rows)[kind]
                x = j + (i - 1) * width
                ax.bar(x, value, width - 0.03, color=color, edgecolor=SURFACE, linewidth=1.5)
                ax.text(x, value + 0.3, str(value), color=TEXT, fontsize=8, ha="center")
        ax.set_xticks(range(len(SETS)))
        ax.set_xticklabels([f"{n}\n({len(eval_rows('q0', s)) if s != 'agent-contexts' else 48})"
                            for n, s in SETS], color=MUTED, fontsize=9)
    for _label, _cfg, _cfg_r, color in CONFIGS:
        axes[0].bar(0, 0, color=color, label=_label)
    fig.legend(frameon=False, loc="lower center", ncol=3, labelcolor=TEXT, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "c2-questions-quality.svg", facecolor=SURFACE)
    plt.close(fig)


def category_chart():
    """Errors per category (english + italian together), per configuration."""
    cats = {}
    for _label, cfg, _cfg_r, _color in CONFIGS:
        for set_name in ("privacy-plus-english", "privacy-plus-italian"):
            for r in eval_rows(cfg, set_name):
                c = cats.setdefault(r["category"], {"expect": r["expect"]})
                c[cfg] = c.get(cfg, 0) + (r["decision"] != r["expect"])
    shown = [k for k, v in cats.items() if any(v.get(c[1], 0) for c in CONFIGS)]
    shown.sort(key=lambda k: (cats[k]["expect"], k))
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    style(ax, "Errors per category, english + italian (leaks: expect local; FP: expect sota)",
          "cases")
    width = 0.26
    for i, (label, cfg, _cfg_r, color) in enumerate(CONFIGS):
        xs = [j + (i - 1) * width for j in range(len(shown))]
        ax.bar(xs, [cats[k].get(cfg, 0) for k in shown], width - 0.03, color=color,
               edgecolor=SURFACE, linewidth=1.5, label=label)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([f"{k}\n({'leak' if cats[k]['expect'] == 'local' else 'FP'})"
                        for k in shown], color=MUTED, fontsize=8, rotation=0)
    fig.legend(frameon=False, loc="lower center", ncol=3, labelcolor=TEXT, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "c2-questions-categories.svg", facecolor=SURFACE)
    plt.close(fig)


PROB = re.compile(r"systemone/(?:llm@([0-9.]+)\((\w+)\)|(\w+)@([0-9.]+)\((?:shown)\))")


def top_probability(signals):
    """Highest probability of a positive question, or None when C2 did not run."""
    probs = []
    for m in PROB.finditer(signals):
        probs.append(float(m.group(1) or m.group(4)))
    return max(probs) if probs else None


def probability_chart():
    """Top C2 probability per case where C2 ran, split by the expected decision."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor=SURFACE, sharey=True)
    bins = [i / 20 for i in range(21)]
    for ax, (_label, cfg, cfg_r, _color) in zip(axes, CONFIGS, strict=True):
        short = {"q0": "Q0: today", "q1": "Q1: split, with C1", "q2": "Q2: split, without C1"}
        style(ax, short[cfg], "cases (log scale)" if cfg == "q0" else "")
        local, sota = [], []
        for set_name in ("privacy-plus-english", "privacy-plus-italian", "agent-contexts"):
            for r in eval_rows(cfg_r if set_name == "agent-contexts" else cfg, set_name):
                p = top_probability(r["signals"])
                if p is not None:
                    (local if r["expect"] == "local" else sota).append(p)
        ax.hist([sota, local], bins=bins, color=[AQUA, ORANGE], stacked=False,
                label=["expect sota", "expect local"], edgecolor=SURFACE)
        ax.axvline(0.5, color=TEXT, linewidth=1)
        ax.set_yscale("log")
        ax.text(0.51, 300, "threshold 0.5", color=TEXT, fontsize=8)
        ax.set_xlabel("highest probability of a positive question", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, labelcolor=TEXT,
               fontsize=9)
    fig.suptitle("Highest C2 probability per case where C2 ran (english, italian, agent contexts)",
                 x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "c2-questions-probabilities.svg", facecolor=SURFACE)
    plt.close(fig)


def jsonl(tag, scenario):
    path = RESULTS / f"{PREFIX}-{tag}-{scenario}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def line(ax, xs, ys, color, label, dy=0.0, **kw):
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.2, **kw)
    ax.text(xs[-1], ys[-1] * (1 + dy), f"  {label}", color=TEXT, fontsize=8, va="center")


NSEQ = ((4, BLUE), (8, ORANGE), (16, AQUA), (32, PURPLE))


def concurrency_chart():
    """Throughput and p50 latency by parallel C2 calls, for each --max-num-seqs value
    (split questions v1)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    style(axes[0], "Throughput by parallel C2 calls", "calls per second")
    style(axes[1], "Latency p50 by parallel C2 calls", "milliseconds (log scale)")
    for n, color in NSEQ:
        tag = "nseq4-v1" if n == 4 else f"nseq{n}"
        rows = {r["concurrency"]: r for r in jsonl(tag, "concurrency")
                if r["questions"] == "split-v1"}
        # 32: the first run had two outliers at 2 and 4 calls; the repeat run (up to 32
        # calls) replaces those levels.
        if n == 32:
            rows.update({r["concurrency"]: r for r in jsonl("nseq32-repeat", "concurrency")})
        rows = sorted(rows.values(), key=lambda r: r["concurrency"])
        if not rows:
            continue
        xs = [r["concurrency"] for r in rows]
        line(axes[0], xs, [r["rps"] for r in rows], color, f"max-num-seqs {n}")
        line(axes[1], xs, [r["p50"] for r in rows], color, f"{n}")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
        ax.set_xlabel("parallel calls (short prompts, about 1,000 tokens)", color=MUTED)
        ax.set_xlim(0.8, 130)
    axes[1].set_yscale("log")
    ticks = [100, 200, 500, 1000, 2000, 3000]
    axes[1].set_yticks(ticks)
    axes[1].set_yticklabels([f"{t:,}" for t in ticks])
    axes[1].minorticks_off()
    fig.tight_layout()
    fig.savefig(OUT / "c2-max-num-seqs.svg", facecolor=SURFACE)
    plt.close(fig)


def questions_cost_chart():
    """p50 latency by prompt size: built-in questions against split questions v1."""
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax, "Cost of the split questions: C2 latency by prompt size (one call)", "seconds")
    rows = jsonl("nseq4", "size") + jsonl("nseq4-v1", "size")
    for q, color, label in (("built-in", BLUE, "built-in (3 questions)"),
                            ("split-v1", ORANGE, "split v1 (9 questions)")):
        sel = sorted((r for r in rows if r["questions"] == q), key=lambda r: r["tokens"])
        if sel:
            line(ax, [r["tokens"] / 1000 for r in sel], [r["p50"] / 1000 for r in sel], color,
                 label, dy=0.08 if q == "split-v1" else -0.08)
    ax.set_xlabel("prompt tokens, thousands", color=MUTED)
    ax.set_xlim(0, 40)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "c2-questions-cost.svg", facecolor=SURFACE)
    plt.close(fig)


def gate_chart():
    """Privacy gate time on agent contexts: with C1 (Presidio + C2) and without C1 (C2 only),
    cold texts and growing agent turns (prefix cache); and parallel agents."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    style(axes[0], "Gate time by agent context size (one agent)", "seconds")
    ctx = jsonl("nseq4", "context")
    chars = {r["size"]: r["chars"] for r in ctx if r["backend"] == "presidio" and "chars" in r}
    pres = {r["size"]: r["p50"] / 1000 for r in ctx if r["backend"] == "presidio" and "p50" in r}
    c2 = {r["size"]: r["p50"] / 1000 for r in ctx if r["backend"] != "presidio" and "p50" in r}
    both = sorted(s for s in pres if s in c2 and s in chars)
    xs = [chars[s] / 1000 for s in both]
    line(axes[0], xs, [pres[s] + c2[s] for s in both], BLUE, "with C1: Presidio + C2 (cold)")
    line(axes[0], xs, [c2[s] for s in both], ORANGE, "without C1: C2 only (cold)")
    grow = [r for r in jsonl("nseq4", "growth") if r.get("kind") == "grow"]
    if grow:
        grow.sort(key=lambda r: r["chars"])
        line(axes[0], [r["chars"] / 1000 for r in grow], [r["secs"] for r in grow], AQUA,
             "C2, growing agent turns (prefix cache)")
    axes[0].axvline(150, color=MUTED, linewidth=1)
    axes[0].text(148, axes[0].get_ylim()[1] * 0.95, "SOTA size cap\n150,000 chars", color=MUTED,
                 fontsize=8, ha="right", va="top")
    axes[0].set_xlabel("context size, thousands of characters (log text)", color=MUTED)
    axes[0].set_ylim(bottom=0)
    style(axes[1], "Parallel agents, new contexts of about 33k tokens", "seconds, p95")
    par4, par32 = jsonl("nseq4", "parallel"), jsonl("nseq32", "parallel")
    for rows, backend, color, label in (
        (par4, "presidio", BLUE, "Presidio (one worker)"),
        (par4, "systemone-s1", ORANGE, "C2, max-num-seqs 4"),
        (par32, "systemone-s1", PURPLE, "C2, max-num-seqs 32"),
    ):
        sel = sorted((r for r in rows if r["backend"] == backend and "p95" in r),
                     key=lambda r: r["concurrency"])
        if sel:
            line(axes[1], [r["concurrency"] for r in sel], [r["p95"] / 1000 for r in sel], color,
                 label, dy=-0.06 if "32" in label else 0.0)
    # Effective timeouts of the gitops policy for one context of about 71,000 characters:
    # Presidio min(max(3.0, 0.052 x 71), 10) = 3.7 s; C2 min(max(8.0, 0.11 x 71), 15) = 8 s.
    for secs, label, color in ((3.7, "Presidio timeout at this size: 3.7 s", BLUE),
                               (8.0, "C2 timeout at this size: 8 s", ORANGE)):
        axes[1].axhline(secs, color=color, linewidth=1, linestyle=(0, (4, 3)))
        axes[1].text(10.9, secs + 0.4, label, color=MUTED, fontsize=8, ha="right")
    axes[1].set_xticks([1, 4, 8])
    axes[1].set_xlabel("agents at the same time", color=MUTED)
    axes[1].set_xlim(0.5, 11)
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "c2-gate-c1.svg", facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams["svg.fonttype"] = "none"  # text stays text in the SVG
    plt.rcParams["svg.hashsalt"] = "sovereign-selfheal"  # stable ids: same input, same file
    OUT.mkdir(parents=True, exist_ok=True)
    quality_chart()
    category_chart()
    probability_chart()
    concurrency_chart()
    questions_cost_chart()
    gate_chart()


if __name__ == "__main__":
    main()
