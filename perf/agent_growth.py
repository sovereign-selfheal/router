#!/usr/bin/env python3
"""Agent-like growing context for C2: does the vLLM prefix cache help?

One session per backend: a base context, then turns that APPEND new tool output (as an agent
does, and as the router's _whole_payload keeps the order). Each turn is timed. Controls in the
same run: a cold call (fresh text) at the start and end sizes, and an exact repeat of the base.
Only sends requests. Runs in a LiteLLM pod next to c2_perf.py (imports it) and
privacy_scoring.py, like the context scenario of c2_perf.py:

  python3 agent_growth.py \
      --decision-url http://dgemma-decision-predictor.local-models.svc.cluster.local/v1 \
      --chat-url http://qwen38-local-predictor.local-models.svc.cluster.local/v1 \
      --chat-model qwen38-local
"""
import argparse
import json
import time

import c2_perf as P


def emit(rec):
    print(json.dumps(rec), flush=True)


def timed(b, backend, text):
    t0 = time.perf_counter()
    try:
        secs, toks, p = b.call(backend, text)
        p = p if isinstance(p, bool) else round(p, 2)
        return {"secs": round(secs, 2), "tokens": toks, "p": p}
    except Exception as exc:
        return {"secs": round(time.perf_counter() - t0, 2), "error": type(exc).__name__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-url", required=True)
    ap.add_argument("--chat-url", required=True)
    ap.add_argument("--chat-model", required=True)
    ap.add_argument("--backends", default="systemone-s1,chat")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--base", type=int, default=32000, help="size units of c2_perf.agent_text")
    ap.add_argument("--step", type=int, default=4000)
    ap.add_argument("--turns", type=int, default=8)
    args = ap.parse_args()
    b = P.Backends(args)
    for backend in args.backends.split(","):
        b.call(backend, P.text_of(50))  # warm-up
        base = P.agent_text(args.base)
        final_size = args.base + args.step * args.turns
        emit({"backend": backend, "kind": "cold", "size": args.base, "chars": len(base),
              **timed(b, backend, P.agent_text(args.base))})
        emit({"backend": backend, "kind": "first", "turn": 0, "chars": len(base),
              **timed(b, backend, base)})
        emit({"backend": backend, "kind": "repeat", "turn": 0, "chars": len(base),
              **timed(b, backend, base)})
        text = base
        for turn in range(1, args.turns + 1):
            text = text + "\n" + P.agent_text(args.step)
            emit({"backend": backend, "kind": "grow", "turn": turn, "chars": len(text),
                  **timed(b, backend, text)})
        cold = P.agent_text(final_size)
        emit({"backend": backend, "kind": "cold", "size": final_size, "chars": len(cold),
              **timed(b, backend, cold)})
    emit({"done": True})


if __name__ == "__main__":
    main()
