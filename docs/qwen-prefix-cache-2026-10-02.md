# Prefix cache for the local Qwen3.8 model, 2026-10-02

The local model Qwen3.8 27B NVFP4 ran with the vLLM prefix cache off. Each agent turn read its whole
context again: 23 s for 120,000 tokens, also when only the last part was new
([`sota-size-cap-2026-10-02.md`](sota-size-cap-2026-10-02.md), section 2). This page documents the
change that turns the cache on (gitops, `localModel.profiles.gpu.extraArgs`) and the measurements
before and after it, on the same cluster with the same tests.

> **Support status:** LiteLLM and Presidio are community software, not supported by Red Hat. vLLM
> marks the prefix cache for Mamba-like layers as experimental ("Its support for Mamba layers is
> experimental"), also in the Red Hat build used here.

## Change

| Flag | Before | After | Why |
|---|---|---|---|
| `--enable-prefix-caching` | not set (vLLM default: off for this model) | set | Reuse the prefill of a repeated prefix |

Nothing else changes: image, resources, `--max-model-len`, MTP speculative decoding
(`--speculative-config`, 3 tokens), CPU profile.

How vLLM 0.24 (0.24.0+rhaiv.13) handles this hybrid model (16 attention layers, 48 Gated DeltaNet
layers), read in the code of the image and in the logs:

- The model class (`Qwen3_5ForConditionalGeneration`) does not support Mamba cache mode `all`. With
  the flag, vLLM selects mode **`align`** by itself (log: "Mamba cache mode is set to 'align'"). In
  this mode it keeps the Gated DeltaNet state only at block boundaries, and it schedules the prefill
  in whole blocks. No `--mamba-cache-mode` flag is needed. Align mode needs chunked prefill (on) and
  a block size up to `max_num_batched_tokens` (1600 <= 8192): both are true.
- **Block size: 1600 tokens**, before and after (log: "Setting attention block size to 1600 tokens").
  vLLM makes one attention page as large as one Gated DeltaNet state. The KV cache is FP8 (from the
  checkpoint) and the state is float32. The cache reuses whole blocks only.
- **The state stays float32.** `--mamba-ssm-cache-dtype=bfloat16` would make the block about 800
  tokens. We did not set it: the model config asks for `mamba_ssm_dtype: float32`, vLLM says that only
  float32 has no known accuracy issues, and the gain is at most about 800 tokens of prefill per
  request (about 0.1 s).
- `--calculate-kv-scales` is not set (it breaks hybrid models).
- **KV cache size:** 1,472,468 tokens before, 1,426,928 after (-3.1%; vLLM adds 3 padding layers).
  Maximum concurrency at 131,072 tokens: 11.23x before, 10.89x after.

## Setup

- Cluster ocp.48hlt, router v0.8.0. Before: gitops `main` 21a883c. After: gitops branch
  `qwen-prefix-cache` (the same tree plus the flag). Qwen3.8 27B NVFP4 on one RTX PRO 6000 (g7e.2xlarge).
- The same script for both phases, [`perf/qwen_cache_run.sh`](../perf/qwen_cache_run.sh), run from a
  LiteLLM pod. One request at a time; Qwen had no other traffic (0 running, 0 waiting before each run).
- Three tests, each one between two snapshots of the vLLM counters
  ([`perf/vllm_metrics.py`](../perf/vllm_metrics.py)):
  1. **Growing context** ([`perf/agent_growth.py`](../perf/agent_growth.py), chat backend): the C2
     classifier prompt with a base context of about 142,000 characters, then 8 turns of about 18,000
     characters each. Controls: new text at the first and the last size, and the base sent twice.
  2. **Shared fixed prefix** ([`perf/shared_prefix.py`](../perf/shared_prefix.py)): a system prompt
     shaped like the triage agent's, an OpenAI tool schema of 7 tools (like its MCP tools) and runbook
     lines as padding, at 5 target sizes. Per size: 1 cold request, then 5 requests with the same
     prefix and a different user message. Streaming, thinking off; time to first token (TTFT).
  3. **Long answers** (`perf/shared_prefix.py --decode 8`): 8 answers of 1,024 tokens, temperature 0,
     the same prompts in both phases. Decode speed and MTP acceptance.
- Raw data: `perf/results/2026-10-02-ocp.48hlt-qwen-{before,after}-{growth,prefix,decode,metrics,metrics-diff}.jsonl`.

## 1. Growing context

| Call | Tokens | Before | After |
|---|---|---|---|
| New text | 60k | 8.1 s | 8.2 s |
| Base, first time | 60k | 8.0 s | 8.1 s |
| Base again | 60k | 8.0 s | **0.68 s** |
| Turn 1 | 67k | 9.6 s | **2.2 s** |
| Turn 2 | 75k | 11.2 s | **2.2 s** |
| Turn 3 | 82k | 12.9 s | **2.6 s** |
| Turn 4 | 90k | 14.8 s | **2.7 s** |
| Turn 5 | 97k | 16.7 s | **2.8 s** |
| Turn 6 | 105k | 18.9 s | **2.8 s** |
| Turn 7 | 113k | 21.0 s | **3.2 s** |
| Turn 8 | 120k | 23.3 s | **3.2 s** |
| New text | 120k | 23.5 s | 23.1 s |

vLLM counters over the run: prefix cache hit rate 0% before, **69.5%** after. Mean prefill tokens
computed per request: 81,738 before, 24,642 after. Mean TTFT: 13.7 s before, 4.8 s after.

- A turn now costs about 2-3 s instead of 10-23 s. It grows slowly with the context: the new part is
  always about 7,500 tokens, and attention over the cached part still costs time.
- New text costs the same as before (8.2 s and 23.1 s): the cache does not slow down a cold prefill.
- The same text again takes 0.68 s, not 0: the last block (up to 1,600 tokens) is computed again,
  and the answer (up to 200 tokens) is generated.

## 2. Shared fixed prefix

TTFT, median of the 5 warm requests; the cold request in brackets.

| Prompt tokens | Full blocks of 1,600 | Before | After |
|---|---|---|---|
| ~1,500 | 0 | 0.15 s (0.15) | 0.21 s (0.15) |
| ~1,750 | 1 | 0.25 s (0.17) | 0.22 s (0.17) |
| ~3,350 | 2 | 0.38 s (0.30) | **0.18 s** (0.39) |
| ~8,000 | 4-5 | 0.68 s (0.68) | **0.22 s** (0.73) |
| ~31,700 | 19 | 3.38 s (3.38) | **0.46 s** (3.42) |

vLLM counters over the run: hit rate 0% before, 49.6% after (the cold and the calibration requests
count as misses).

- Under about 3,000 tokens there is no gain we can see. TTFT has a floor of about 0.15 s, and in both
  phases it moves between about 0.15 and 0.25 s from one request to the next. A prompt under 1,600
  tokens has no full block, so nothing of it is cached.
- From 2 blocks up, a warm request stays near the floor: 0.18 s at 3,350 tokens, 0.22 s at 8,000,
  0.46 s at 31,700 (7x faster).
- **Advice for the agents:** the fixed part (system prompt + tool schema) is cached only in whole
  blocks of 1,600 tokens. Our test prefix without padding is about 880 tokens: alone it is under one
  block, so different agent sessions do not share it. Inside one session the growing context is
  cached anyway (section 1). To share the fixed part between sessions, it must end just after a
  multiple of 1,600 tokens. The real size of the triage agent's prefix (OGX adds the MCP tool schemas)
  was not measured.

## 3. Long answers and MTP

| | Before | After |
|---|---|---|
| Decode speed, median of 8 answers | 109.5 tokens/s | 112.1 tokens/s |
| MTP acceptance (accepted / draft tokens) | 71.7% | 74.7% |
| MTP mean acceptance length (tokens per step) | 3.15 | 3.24 |

The difference is inside the run-to-run noise (single answers: 105-118 tokens/s before, 107-124
after). The prefix cache and align mode do not slow down MTP. In the other two tests the acceptance
was also the same (87.7% and 86.7% in test 1, 80.1% and 78.0% in test 2).

## 4. Validation

Validation repo, the same groups before and after on ocp.48hlt. Logs:
`pr-texts/validate-2026-10-02-ocp.48hlt-qwen-{before,after}-{default,routing}.log` (workspace, not in git).

| Groups | Before | After |
|---|---|---|
| default | 49/50, WARN P4 | 48/50, WARN P4, **FAIL P3** |
| `routing`, `demo`, `context` | 28/28 | 28/28 |

- **P3 is not caused by this change.** Between the two runs (15:36 UTC) someone installed the MCP
  Gateway Operator v0.7.1 (Tech Preview) in a new namespace `mcp-gateway-system`. OLM created four
  dependency Subscriptions there (rhcl, authorino, limitador, dns operator) with `Automatic`
  approval. P3 checks every Subscription of these packages, so it fails. The demo's own Subscriptions
  in `openshift-operators` are still `Manual`, and the CSVs in `mcp-gateway-system` are copies of
  them (same pinned versions).
- R8 (agent tool call on the local model) passes: tool calls work with the cache on.
- The K cases (agent contexts of 8k-100k tokens) take the same time: each case sends new text, and
  most of the time goes to Presidio and C2, not to the prefill of Qwen.

## Findings

1. **The prefix cache works on Qwen3.8 with vLLM 0.24,** in align mode, together with MTP. A growing
   agent context costs 2-3 s per turn instead of 10-23 s at 60,000-120,000 tokens.
2. **Nothing gets slower:** cold prefill, decode speed and MTP acceptance are the same. The KV cache
   is 3.1% smaller.
3. **Small prompts gain nothing:** the cache reuses blocks of 1,600 tokens. The C2 chat fallback and
   short agent calls see no change.
4. This also helps C2 when it falls back to Qwen: the C2 chat prompt of a growing context is now
   cached too (test 1 uses exactly that prompt).

## Not measured

- Several agents at once: how long cached blocks stay in the KV cache under load (eviction).
- The real triage agent loop end to end (OGX through the router).
- The bfloat16 state (`--mamba-ssm-cache-dtype=bfloat16`, block of about 800 tokens) and its accuracy.

## Reproduce

```bash
# From a LiteLLM pod: copy perf/c2_perf.py, perf/agent_growth.py, perf/shared_prefix.py,
# perf/vllm_metrics.py, perf/qwen_cache_run.sh and litellm/privacy_scoring.py to /tmp/perf
cd /tmp/perf && sh qwen_cache_run.sh before     # or: after
```
