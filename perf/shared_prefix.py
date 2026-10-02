#!/usr/bin/env python3
"""Shared fixed prefix: does the vLLM prefix cache of the local chat model reuse it?

An agent sends the same system prompt and tool schema at the start of every request; only the
messages after them change. This script builds such a prefix (the triage-agent system prompt, an
OpenAI tool schema like its MCP tools, and runbook lines as padding) at several target sizes in
tokens. For each size it sends one cold request, then requests with the same prefix and a different
user message. It measures the time to the first token (streaming) and the total time.

The prefix cache reuses whole blocks only (vLLM logs "Setting attention block size to N tokens";
N = 1600 for Qwen3.8 on 2026-10-02). Pick sizes just below and just above multiples of N to see it.
Each size starts with its own random session line, so no size reuses the cache of another one.

Only sends requests. Runs in a LiteLLM pod next to c2_perf.py (imports it), like agent_growth.py:

  python3 shared_prefix.py \
      --chat-url http://qwen38-local-predictor.local-models.svc.cluster.local/v1 \
      --chat-model qwen38-local --sizes 1500,1700,3300,8000,32000

With --decode N it sends N long answers instead (same prompts every run, temperature 0), to compare
the decode speed and the MTP acceptance before and after a change.
"""
import argparse
import json
import random
import time
import urllib.request

import c2_perf as P

SYSTEM_PROMPT = """You are an OpenShift SRE troubleshooting assistant. Diagnose application \
problems using the tools below, then open an incident ticket and report your findings.

Workflow:
1. List pods in the target namespace (default: agentic-triage). Note status and restarts.
2. For unhealthy pods, use pods_get to check events and conditions.
3. Query Prometheus for error rates and latency.
4. Synthesize a diagnosis from the data you collected.
5. Call create_incident with a short description, your full diagnosis, impact and urgency.
6. Output your diagnosis report.
"""


def tool(name, description, props, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


STR = {"type": "string"}
INT = {"type": "integer"}
TOOLS = [
    tool("query_prometheus", "Run an instant PromQL query and return the result as JSON.",
         {"promql": STR}, ["promql"]),
    tool("query_prometheus_range", "Run a PromQL range query over the last minutes.",
         {"promql": STR, "duration_minutes": INT}, ["promql"]),
    tool("create_incident", "Open an incident ticket in the ticketing system.",
         {"short_description": STR, "description": STR, "impact": INT, "urgency": INT,
          "category": STR},
         ["short_description", "description"]),
    tool("add_work_note", "Add a work note to an existing incident.",
         {"number": STR, "note": STR}, ["number", "note"]),
    tool("pods_list_in_namespace", "List the pods of a namespace with status and restarts.",
         {"namespace": STR}, ["namespace"]),
    tool("pods_get", "Get one pod with its events and conditions.",
         {"namespace": STR, "name": STR}, ["namespace", "name"]),
    tool("nodes_top", "Show CPU and memory use of the nodes.", {}, []),
]

QUESTIONS = (
    "The checkout service returns HTTP 503 since 10 minutes. Find the cause.",
    "Pods of quarkus-buggy-app restart often in agentic-triage. Why?",
    "Latency of the orders endpoint doubled after the last rollout. Investigate.",
    "Users report failed payments. Check error rates and open a ticket.",
    "A node shows MemoryPressure. Which workloads are affected?",
    "The error rate of /api/cart is above 5%. Diagnose it.",
    "Check the health of the namespace agentic-triage and report.",
    "An alert says the payment pod is in CrashLoopBackOff. Find the root cause.",
)


def runbook(lines, seed):
    """Deterministic runbook lines (same seed, same text): the padding of the fixed prefix."""
    rnd = random.Random(seed)
    out = []
    for _ in range(lines):
        out.append(rnd.choice(P.AGENT_LINES).format(
            ts=f"2026-10-01T{rnd.randrange(24):02d}:{rnd.randrange(60):02d}:00Z",
            n=rnd.randrange(10, 9999), n2=rnd.randrange(1, 99), p=rnd.randrange(1, 4),
            hexid=f"{rnd.getrandbits(64):016x}"))
    return "Known past incidents (runbook):\n" + "\n".join(f"- {line}" for line in out)


def messages(session, lines, question):
    system = f"Session {session}.\n{SYSTEM_PROMPT}\n{runbook(lines, session)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def stream(args, msgs, tools=True):
    """One streaming request. Returns time to first token, total time and the usage block."""
    body = {"model": args.chat_model, "max_tokens": args.max_tokens, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": msgs}
    if tools:
        body.update({"tools": TOOLS, "tool_choice": "auto"})
    url = args.chat_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.perf_counter()
    ttft, usage = None, {}
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")
                                     or delta.get("reasoning") or delta.get("tool_calls")):
                    ttft = time.perf_counter() - t0
    return ttft, time.perf_counter() - t0, usage


def prompt_tokens(args, lines):
    """Prompt tokens of a prefix with `lines` runbook lines (fresh session, so nothing is shared
    with the measured requests)."""
    _, _, usage = stream(args, messages(random.getrandbits(48), lines, QUESTIONS[0]))
    return usage.get("prompt_tokens", 0)


def lines_for(args, target):
    """Number of runbook lines for a prompt of about `target` tokens (linear fit, two calls)."""
    t0 = prompt_tokens(args, 0)
    probe = max(1, (target - t0) // 25)
    t1 = prompt_tokens(args, probe)
    per_line = max(1.0, (t1 - t0) / probe)
    return max(0, round((target - t0) / per_line)), t0


def emit(rec):
    print(json.dumps(rec), flush=True)


def decode(args):
    """Long answers with the same prompts before and after a change (temperature 0): decode speed
    and, with vllm_metrics.py around the run, the MTP acceptance."""
    for i in range(args.decode):
        # No tools and no agent prompt: with them the model answers with a short tool call.
        system = "You write clear incident reports.\n" + runbook(20, i)
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": "Write a post-incident report of about 600 words about "
                 "the past incidents above: timeline, root causes, impact and follow-up actions."}]
        rec = {"kind": "decode", "i": i}
        try:
            ttft, secs, usage = stream(args, msgs, tools=False)
            out = usage.get("completion_tokens") or 0
            rec.update({"ttft": round(ttft, 3) if ttft is not None else None,
                        "secs": round(secs, 3), "completion_tokens": out,
                        "tok_per_s": round(out / (secs - (ttft or 0)), 1) if out else None})
        except Exception as exc:
            rec["error"] = type(exc).__name__
        emit(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-url", required=True)
    ap.add_argument("--chat-model", required=True)
    ap.add_argument("--sizes", default="1500,1700,3300,8000,32000",
                    help="target prompt sizes in tokens")
    ap.add_argument("--warm", type=int, default=5, help="requests after the cold one, per size")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--decode", type=int, default=0,
                    help="only run N long answers (max_tokens 1024) instead of the prefix test")
    args = ap.parse_args()
    stream(args, messages(random.getrandbits(48), 0, QUESTIONS[0]))  # warm-up
    if args.decode:
        args.max_tokens = 1024
        decode(args)
        emit({"done": True})
        return
    for target in (int(x) for x in args.sizes.split(",")):
        lines, base_tokens = lines_for(args, target)
        session = random.getrandbits(48)
        for i in range(args.warm + 1):
            q = QUESTIONS[i % len(QUESTIONS)]
            rec = {"target": target, "lines": lines, "base_tokens": base_tokens,
                   "kind": "cold" if i == 0 else "warm", "i": i}
            try:
                ttft, secs, usage = stream(args, messages(session, lines, q))
                details = usage.get("prompt_tokens_details") or {}
                rec.update({"ttft": round(ttft, 3) if ttft is not None else None,
                            "secs": round(secs, 3), "prompt_tokens": usage.get("prompt_tokens"),
                            "cached_tokens": details.get("cached_tokens")})
            except Exception as exc:
                rec["error"] = type(exc).__name__
            emit(rec)
    emit({"done": True})


if __name__ == "__main__":
    main()
