#!/usr/bin/env python3
"""Charts of docs/sota-size-cap-2026-10-01.md, from the raw data in perf/results/.

    uv run --with matplotlib==3.10.7 perf/plot_context_cap.py

Writes three SVG files in docs/img/. The data files are the output of
`c2_perf.py --scenarios context` and the K1-K10 results of the validation repo.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "perf/results/2026-10-01-ocp.5bw8q-v080-context.jsonl"
VALIDATION = ROOT / "perf/results/2026-10-01-ocp.5bw8q-v080-validation.json"
OUT = ROOT / "docs/img"

# Validated categorical slots 1-3 (blue, orange, aqua) and neutral text colors.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

CAP = 117000  # efficiency.sota_max_prompt_chars
PRESIDIO = (3.0, 0.052, 10.0)  # timeout_seconds, timeout_per_1k_chars, timeout_max_seconds
C2 = (8.0, 0.11, 15.0)


def timeout(cfg, chars):
    base, per_1k, top = cfg
    return min(max(base, per_1k * chars / 1000), max(base, top))


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


def load_perf():
    """Characters (from the Presidio row of the same size) and p50 seconds per backend."""
    rows = [json.loads(line) for line in PERF.read_text().splitlines() if line.startswith("{")]
    chars = {r["size"]: r["chars"] for r in rows if r["backend"] == "presidio"}
    series = {}
    for r in rows:
        series.setdefault(r["backend"], []).append((chars[r["size"]], r["p50"] / 1000))
    return series


def timeout_chart(name, title, measured, cfg, old):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax, title, "seconds")
    xs = range(0, 150001, 1000)
    ax.plot(
        [x / 1000 for x in xs], [old] * len(xs), color=MUTED, linewidth=1.5, linestyle=(0, (4, 3))
    )
    ax.text(2, old + 0.25, f"fixed timeout before ({old:g} s)", color=MUTED, fontsize=9)
    ax.plot([x / 1000 for x in xs], [timeout(cfg, x) for x in xs], color=TEXT, linewidth=2)
    ax.text(150, timeout(cfg, 150000) + 0.25, "timeout v0.8.0", color=TEXT, fontsize=9, ha="right")
    for (label, color, points), dy in zip(measured, (0.0, -0.6), strict=False):
        x, y = zip(*sorted(points), strict=True)
        ax.plot(
            [v / 1000 for v in x],
            y,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.text(x[-1] / 1000 + 1.5, y[-1] + dy, label, color=TEXT, fontsize=9, va="center")
    ax.axvline(CAP / 1000, color=MUTED, linewidth=1)
    ax.text(
        CAP / 1000 - 1,
        ax.get_ylim()[1] * 0.97,
        "SOTA size cap\n117,000 chars",
        color=MUTED,
        fontsize=9,
        ha="right",
        va="top",
    )
    ax.set_xlim(0, 165)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(
        "request size, thousands of characters (log text, about 2.25 per token)", color=MUTED
    )
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)


def first_chunk_chart(phases):
    p1 = {c["id"]: c for c in phases["phase1"]["cases"]}
    p2 = {c["id"]: c for c in phases["phase2"]["cases"]}
    ids = [c["id"] for c in phases["phase2"]["cases"]]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    style(ax, "Time to the first chunk, agent contexts K1-K10 (streamed, LOCAL answer)", "seconds")
    width = 0.38
    for i, rid in enumerate(ids):
        for off, case, color in ((-width / 2, p1.get(rid), BLUE), (width / 2, p2.get(rid), ORANGE)):
            if not case or case["first_chunk_s"] is None:
                continue
            ax.bar(
                i + off,
                case["first_chunk_s"],
                width - 0.04,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
            )
        c2 = p2[rid]
        if c2["decided_by"] == "efficiency":
            ax.text(
                i + width / 2, c2["first_chunk_s"] + 0.3, "cap", color=TEXT, fontsize=8, ha="center"
            )
        if rid in p1 and p1[rid]["status"] == "SKIP":
            ax.text(i - width / 2, 0.3, "skip", color=MUTED, fontsize=8, ha="center")
        if rid in p1 and "error" in (p1[rid]["reason"] or ""):
            ax.text(
                i - width / 2,
                p1[rid]["first_chunk_s"] + 0.3,
                "timeout",
                color=TEXT,
                fontsize=8,
                ha="center",
            )
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(
        [f"{rid}\n{p2[rid]['prompt_chars'] // 1000}k ch" for rid in ids], color=MUTED, fontsize=9
    )
    ax.bar(0, 0, color=BLUE, label="keys off (as v0.7.0)")
    ax.bar(0, 0, color=ORANGE, label="keys on")
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(OUT / "first-chunk-k1-k10.svg", facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams["svg.fonttype"] = "none"  # text stays text in the SVG
    plt.rcParams["svg.hashsalt"] = "sovereign-selfheal"  # stable ids: same input, same file
    OUT.mkdir(parents=True, exist_ok=True)
    perf = load_perf()
    timeout_chart(
        "presidio-timeout.svg",
        "Presidio /analyze: measured time and timeout",
        [("Presidio p50", BLUE, perf["presidio"])],
        PRESIDIO,
        PRESIDIO[0],
    )
    timeout_chart(
        "c2-timeout.svg",
        "C2 classifier: measured time and timeout (per call)",
        [
            ("systemone p50", ORANGE, perf["systemone-s1"]),
            ("chat (Qwen3.8) p50", AQUA, perf["chat"]),
        ],
        C2,
        C2[0],
    )
    first_chunk_chart(json.loads(VALIDATION.read_text())["phases"])


if __name__ == "__main__":
    main()
