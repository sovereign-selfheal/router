#!/usr/bin/env python3
"""Snapshot of the vLLM counters that show the prefix cache and the MTP speculative decoding.

Reads the Prometheus text of `<base>/metrics` (the same port as the OpenAI API) and prints one JSON
line with the selected counters. Run it before and after a test: the difference of two snapshots is
the effect of the test. With --diff, it reads two snapshot lines and prints the deltas and the
ratios.
Standard library only; it runs in a LiteLLM pod next to c2_perf.py.

  python3 vllm_metrics.py --url http://qwen38-local-predictor.local-models.svc.cluster.local \
      --label before-growth >> metrics.jsonl
  python3 vllm_metrics.py --diff metrics.jsonl before-growth after-growth
"""
import argparse
import json
import time
import urllib.request

# Counter names without the "_total" suffix; histograms by their _sum and _count.
COUNTERS = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:prompt_tokens",
    "vllm:prompt_tokens_cached",
    "vllm:generation_tokens",
    "vllm:request_success",
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)
HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:e2e_request_latency_seconds",
)


def snapshot(url, timeout=30):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as resp:
        text = resp.read().decode()
    wanted = {f"{c}_total": c for c in COUNTERS}
    for h in HISTOGRAMS:
        wanted[f"{h}_sum"] = f"{h}_sum"
        wanted[f"{h}_count"] = f"{h}_count"
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in wanted:
            # Sum over label sets (finished_reason of request_success, for example).
            key = wanted[name]
            values[key] = values.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    gauges = {}
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            labels = line.split("{", 1)[1].rsplit("}", 1)[0]
            for item in labels.split('",'):
                k, _, v = item.partition('="')
                if k in ("block_size", "enable_prefix_caching", "num_gpu_blocks",
                         "kv_cache_size_tokens", "kv_cache_max_concurrency",
                         "mamba_cache_mode", "mamba_block_size", "mamba_ssm_cache_dtype"):
                    gauges[k] = v.rstrip('"')
    return values, gauges


def diff(path, a, b):
    rows = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            rows[rec["label"]] = rec
    va, vb = rows[a]["values"], rows[b]["values"]
    d = {k: round(vb.get(k, 0.0) - va.get(k, 0.0), 3) for k in vb}
    out = {"from": a, "to": b, "delta": d}
    if d.get("vllm:prefix_cache_queries"):
        out["prefix_cache_hit_rate"] = round(
            d.get("vllm:prefix_cache_hits", 0.0) / d["vllm:prefix_cache_queries"], 4)
    if d.get("vllm:spec_decode_num_draft_tokens"):
        out["mtp_acceptance_rate"] = round(
            d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
            / d["vllm:spec_decode_num_draft_tokens"], 4)
    if d.get("vllm:spec_decode_num_drafts"):
        # Tokens per decode step: 1 (the verified token) + accepted draft tokens per draft.
        out["mtp_mean_acceptance_length"] = round(
            1 + d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
            / d["vllm:spec_decode_num_drafts"], 3)
    for h in HISTOGRAMS:
        if d.get(f"{h}_count"):
            out[f"{h}_mean"] = round(d[f"{h}_sum"] / d[f"{h}_count"], 3)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="vLLM base URL, without /v1")
    ap.add_argument("--label", default="snapshot")
    ap.add_argument("--diff", nargs=3, metavar=("FILE", "FROM", "TO"))
    args = ap.parse_args()
    if args.diff:
        diff(*args.diff)
        return
    values, gauges = snapshot(args.url)
    print(json.dumps({"label": args.label, "ts": round(time.time(), 1),
                      "cache_config": gauges, "values": values}), flush=True)


if __name__ == "__main__":
    main()
