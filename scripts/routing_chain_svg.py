"""Write docs/img/routing-chain.svg: the routing chain of the router, animated (SMIL, no script).

Five example requests run one after the other in a 20 s loop. With prefers-reduced-motion the
moving parts are hidden and a static caption lists the five examples. The image has its own dark
panel (the palette of the routing-live-view page), so it reads the same in the light and the dark
theme of GitHub. Change this file when the chain changes, then run it again:

    python3 scripts/routing_chain_svg.py docs/img/routing-chain.svg
"""

import sys

BG, PANEL, LINE, FG, MUTED = "#0c1320", "#131c2b", "#2a3750", "#e7edf6", "#8d9ab0"
WIRE = "#5b6e92"  # the wires of the chain: visible on the dark panel
ROUTER, LOCAL, EXTERNAL, RESTRICTED = "#7ea6ff", "#34d3b8", "#f4b04a", "#ff6670"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
CYCLE = 20  # seconds for the five scenarios (4 s each)

GATES = [  # (title, question, center x)
    ("1 · Namespace", "restricted label?", 259),
    ("2 · Efficiency", "short or simple?", 391),
    ("3 · Privacy", "PII, rules, C2", 523),
    ("4 · Tiering", "team may use SOTA?", 655),
]
DROPS = [  # path from the gate to the local model, label
    ("M259,216 C259,300 360,300 380,356", "restricted"),
    ("M391,216 C391,290 410,320 415,356", "short / simple"),
    ("M523,216 C523,290 470,320 465,356", "sensitive"),
    ("M655,216 C655,300 520,300 500,356", "team capped"),
]
SCENARIOS = [  # dot path, caption, highlighted gate index (None = SOTA), destination
    ("M104,184 L259,184 L259,216 C259,300 360,300 380,356",
     "Alert from payments (label restricted): the namespace gate keeps it local, "
     "no other gate runs",
     0, "local"),
    ("M104,184 L391,184 L391,216 C391,290 410,320 415,356",
     "“What is the capital of France?”: short and simple, the efficiency gate keeps it local",
     1, "local"),
    ("M104,184 L523,184 L523,216 C523,290 470,320 465,356",
     "A customer record with an IBAN: the privacy gate finds personal data and keeps it local",
     2, "local"),
    ("M104,184 L655,184 L655,216 C655,300 520,300 500,356",
     "Long and clean, but from team legal: every gate passes, "
     "tiering allows legal only the local model",
     3, "local"),
    ("M104,184 L865,184",
     "Long, complex, nothing sensitive (agentic-triage, public): every step passes, "
     "the SOTA model answers",
     None, "sota"),
]


def ft(x):
    return f"{x:.4f}".rstrip("0").rstrip(".") if x else "0"


def window(start, on, off):
    """keyTimes/values for an opacity that is 1 between start+on and start+off (fractions)."""
    a, b = start + on, start + off
    keys = [0.0, a, a + 0.005, b, b + 0.005, 1.0]
    vals = [0, 0, 1, 1, 0, 0]
    if a <= 0:
        keys, vals = [0.0, b, b + 0.005, 1.0], [1, 1, 0, 0]
    return ";".join(ft(k) for k in keys), ";".join(str(v) for v in vals)


def fade(start, on, off):
    keys, vals = window(start, on, off)
    return (f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="linear" keyTimes="{keys}" values="{vals}"/>')


def text(x, y, s, size=12, fill=MUTED, weight="normal", anchor="start", extra=""):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{extra}>{s}</text>')


def build():
    out = []
    add = out.append
    add('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 500" width="960" height="500" '
        f'role="img" aria-labelledby="t d" font-family="{SANS}">')
    add('<title id="t">Routing chain of the router</title>')
    add('<desc id="d">A request passes four steps in order: namespace gate, efficiency gate, '
        'privacy gate, tiering. The first step that says LOCAL sends it to the local GPU model in '
        'the cluster; only a request that passes every step goes to the external SOTA model.'
        '</desc>')
    add("<style>.static{display:none}"
        "@media (prefers-reduced-motion: reduce)"
        "{.anim{display:none}.static{display:inline}}"
        "</style>")
    add('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{WIRE}"/></marker>'
        '<marker id="arrow-local" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{LOCAL}"/></marker>'
        '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/>'
        '<feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
    # panel and titles
    add(f'<rect width="960" height="500" rx="18" fill="{BG}"/>')
    add(text(28, 40, "How the router decides, request by request", 19, FG, "600"))
    add(text(28, 62, "router v0.11.1 · the first step that says LOCAL wins · "
                     "SOTA only when every step lets the request pass", 12.5))
    # cluster boundary
    add(f'<rect x="20" y="84" width="730" height="362" rx="16" fill="none" stroke="{LINE}" '
        'stroke-width="1.5" stroke-dasharray="6 6"/>')
    add(text(36, 106, "OPENSHIFT CLUSTER", 11, MUTED, "600", extra=' letter-spacing="1.2"'))
    add(text(790, 106, "OUTSIDE THE CLUSTER", 11, MUTED, "600", extra=' letter-spacing="1.2"'))
    # request
    add(f'<rect x="40" y="150" width="128" height="68" rx="12" fill="{PANEL}" stroke="{LINE}" '
        'stroke-width="1.5"/>')
    add(text(104, 180, "Request", 15, FG, "600", "middle"))
    add(text(104, 200, "agent or app", 11.5, MUTED, anchor="middle"))
    # router
    add(f'<rect x="188" y="122" width="548" height="124" rx="14" fill="#101a2b" stroke="{ROUTER}" '
        'stroke-opacity="0.55" stroke-width="1.5"/>')
    add(text(204, 141, "LITELLM ROUTER · POLICY HOOK", 11, ROUTER, "600",
             extra=' letter-spacing="1.2"'))
    # wires of the chain (the gap in the boundary first, so that the last wire crosses it)
    add(f'<rect x="746" y="174" width="8" height="20" fill="{BG}"/>')
    for x1, x2 in ((168, 199), (318, 331), (450, 463), (582, 595), (714, 789)):
        add(f'<line x1="{x1}" y1="184" x2="{x2}" y2="184" stroke="{WIRE}" stroke-width="2" '
            'marker-end="url(#arrow)"/>')
    # drops to the local model
    for d, _label in DROPS:
        add(f'<path d="{d}" fill="none" stroke="{LOCAL}" stroke-opacity="0.5" stroke-width="2" '
            'marker-end="url(#arrow-local)"/>')
    for (_, label), (_, _, cx) in zip(DROPS, GATES, strict=True):
        add(text(cx + 8, 234, label, 11.5, LOCAL))
    # gates
    for title, question, cx in GATES:
        add(f'<rect x="{cx - 59}" y="152" width="118" height="64" rx="10" fill="{PANEL}" '
            f'stroke="{LINE}" stroke-width="1.5"/>')
        add(text(cx, 178, title, 13.5, FG, "600", "middle"))
        add(text(cx, 198, question, 11, MUTED, anchor="middle"))
    # destinations
    add(f'<rect x="790" y="150" width="150" height="68" rx="12" fill="{PANEL}" stroke="{EXTERNAL}" '
        'stroke-width="1.5"/>')
    add(text(865, 180, "SOTA model", 15, FG, "600", "middle"))
    add(text(865, 200, "external · Gemini", 11.5, MUTED, anchor="middle"))
    add(f'<rect x="300" y="356" width="260" height="64" rx="12" fill="{PANEL}" stroke="{LOCAL}" '
        'stroke-width="1.5"/>')
    add(text(430, 383, "Local GPU model", 15, FG, "600", "middle"))
    add(text(430, 403, "Qwen3.8 on vLLM · the data stays here", 11.5, MUTED, anchor="middle"))
    add(text(36, 434, "public or no label: the namespace gate lets the request pass", 11, MUTED))
    # animation: one dot, one gate highlight, one destination glow and one caption per scenario.
    # Times in seconds inside the 4 s window of each scenario, written as fractions of the cycle.
    win = CYCLE / len(SCENARIOS)
    add('<g class="anim">')
    for i, (path, caption, gate, dest) in enumerate(SCENARIOS):
        s = i * win / CYCLE

        def at(sec):
            return sec / CYCLE

        color = EXTERNAL if dest == "sota" else LOCAL
        if gate is not None:
            cx = GATES[gate][2]
            add(f'<rect x="{cx - 59}" y="152" width="118" height="64" rx="10" fill="none" '
                f'stroke="{RESTRICTED if gate == 0 else color}" stroke-width="3" opacity="0" '
                f'filter="url(#glow)">{fade(s, at(0.8), at(3.5))}</rect>')
            add(f'<rect x="300" y="356" width="260" height="64" rx="12" fill="none" '
                f'stroke="{LOCAL}" stroke-width="3" opacity="0" filter="url(#glow)">'
                f'{fade(s, at(2.4), at(3.5))}</rect>')
        else:
            add(f'<rect x="790" y="150" width="150" height="68" rx="12" fill="none" '
                f'stroke="{EXTERNAL}" stroke-width="3" opacity="0" filter="url(#glow)">'
                f'{fade(s, at(2.4), at(3.5))}</rect>')
        k = ";".join(ft(x) for x in (0.0, s + at(0.16), s + at(2.4), 1.0))
        add(f'<circle r="7" fill="{FG}" stroke="{color}" stroke-width="3" opacity="0">'
            f'<animateMotion dur="{CYCLE}s" repeatCount="indefinite" calcMode="linear" '
            f'keyPoints="0;0;1;1" keyTimes="{k}" path="{path}"/>'
            f'{fade(s, at(0.08), at(3.5))}</circle>')
        add(f'<g opacity="0">{fade(s, 0.0, at(3.85))}'
            + text(480, 476, f"{i + 1}/{len(SCENARIOS)}  {caption}", 13.5, FG, anchor="middle")
            + "</g>")
    add("</g>")
    add('<g class="static">' + text(
        480, 476, "Examples: restricted namespace \u2192 local \u00b7 short question \u2192 local "
        "\u00b7 personal data \u2192 local \u00b7 team legal \u2192 local \u00b7 long and clean "
        "\u2192 SOTA", 13, FG, anchor="middle") + "</g>")
    add("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        fh.write(build())
