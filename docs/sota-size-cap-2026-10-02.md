# SOTA size cap with gemini-2.5-pro: 150,000 characters, 2026-10-02

On 2026-10-02 the SOTA model changed from `openai/Qwen3.8-27B` (RHDP LiteMaaS) to
`openai/gemini-2.5-pro` (RHDP MaaS). The cap of 117,000 characters
([`sota-size-cap-2026-10-01.md`](sota-size-cap-2026-10-01.md)) came from the 65,536-token window of
the old model. This page explains the new value, 150,000 characters, and documents the measurements
behind it.

> **Support status:** LiteLLM and Presidio are community software, not supported by Red Hat. The
> decision model (`systemone`) runs on an unsupported preview image.

## Value

| Key | Old | New | Why |
|---|---|---|---|
| `efficiency.sota_max_prompt_chars` (`chain.yaml`) | `117000` | `150000` | The SOTA window is no longer the limit. The detectors are: at 150,000 characters Presidio and C2 answer within their current timeouts, with a margin (table below) |

The timeouts do not change (`ner.timeout_per_1k_chars` 0.052, max 10 s;
`classifier.timeout_per_1k_chars` 0.11, max 15 s).

The cap is the smallest of two limits:

1. **The SOTA model:** `(context window - max_tokens of the agents - margin) x characters per token`.
   MaaS reports a window of 1,048,576 input tokens for gemini-2.5-pro (`/model/info`). We tested
   252,407 tokens (410,840 characters of logs, answer in 18 s); larger requests were not tested.
   Gemini counts dense log text at about 1.6 characters per token (Qwen: about 2.3).
2. **The detectors:** the largest text that Presidio and C2 read within their timeouts. This is now
   the lower limit.

## Setup

- Cluster ocp.48hlt, router v0.8.0, gitops `main` 7490592. Local model: Qwen3.8 27B NVFP4 on an RTX
  PRO 6000 (vLLM 0.24.0). C2: decision model (`systemone`, `samples: 1`, L40S) with the chat fallback
  on Qwen3.8. Presidio: 2 replicas, gunicorn 1 worker.
- Text like the agents' tool output: Kubernetes log lines, events and ticket lines (the generator of
  [`perf/c2_perf.py`](../perf/c2_perf.py)).
- One call at a time, from a LiteLLM pod. Only requests were sent; nothing in the cluster changed.

## 1. New text at each call (worst case)

[`perf/c2_perf.py --scenarios context`](../perf/c2_perf.py), 3 calls per size, median. Every call
has new text, so no cache can help. Raw data:
[`perf/results/2026-10-02-ocp.48hlt-context.jsonl`](../perf/results/2026-10-02-ocp.48hlt-context.jsonl).

| Characters | Presidio | Presidio timeout | `systemone` | Qwen (C2 fallback) | C2 timeout |
|---|---|---|---|---|---|
| 143,059 | 5.0 s | 7.4 s | 10.6 s | 8.2 s | 15 s |
| **150,045** | 5.4 s | 7.8 s | 11.7 s | 8.9 s | 15 s |
| 178,512 | 6.2 s | 9.3 s | **15.1 s** | 11.3 s | 15 s |
| 200,708 | 7.2 s | 10 s | **18.8 s** | 13.5 s | 15 s |
| 241,906 | 8.5 s | 10 s | **25.3 s** | **17.6 s** | 15 s |
| 280,811 | **9.9 s** | 10 s | **34.1 s** | **22.8 s** | 15 s |

Bold: at or above the timeout. The 143,059 row repeats the largest size of 2026-10-01 (5.07 s and
10.5 s then): the two clusters give the same numbers.

- Up to about 150,000 characters every detector answers in time: margin 1.44 for Presidio, 1.28 for
  `systemone`.
- From about 179,000 characters `systemone` passes 15 s. The chat fallback answers, but the request
  waits 15 s plus 11-14 s.
- From about 242,000 characters both C2 backends time out. The request stays LOCAL after about 30 s
  (fail closed), so an error decides, not the policy.
- `systemone` grows faster than linearly: twice the text (143k to 281k characters) takes more than 3
  times longer. One `systemone` call is one read of the whole text (`samples: 1`, one denoise step,
  three questions in one read), so the time is almost only prefill, and attention in prefill grows
  with the square of the length.

## 2. A context that grows like an agent's

An agent sends the same context again at each turn, with new tool output at the end. The router keeps
the message order (`_whole_payload`), and the decision server puts the fixed system prompt first and
the text after it. So the start of the prompt does not change between turns, and the vLLM prefix cache
can reuse it.

[`perf/agent_growth.py`](../perf/agent_growth.py): a base context of about 142,000 characters, then 8
turns that each add about 18,000 characters. Controls in the same run: new text at the first and the
last size, and the base sent twice. Raw data:
[`perf/results/2026-10-02-ocp.48hlt-growth.jsonl`](../perf/results/2026-10-02-ocp.48hlt-growth.jsonl).

| Call | Characters | `systemone` | Qwen (C2 fallback) |
|---|---|---|---|
| New text | 142k | 10.5 s | 8.1 s |
| Same text again | 142k | **0.28 s** | 8.0 s |
| Turn 1 | 160k | **2.5 s** | 9.6 s |
| Turn 4 | 213k | **3.1 s** | 14.9 s |
| Turn 7 | 266k | **3.8 s** | 21.3 s |
| Turn 8 | 284k | 7.5 s | 23.7 s |
| New text | 284k | 34.9 s | 23.6 s |

- **The prefix cache works for `systemone`** (`--enable-prefix-caching`). Each turn reads only the
  new part: under 4 s up to 266,000 characters. At 284,000 characters (about 127,000 tokens) it takes
  7.5 s; the cache may not hold the whole context there (not checked).
- **Qwen gets no gain.** The vLLM log of the local model shows `enable_prefix_caching=False` and a
  prefix cache hit rate of 0.0%. Qwen3.8 is a hybrid model (attention and Mamba-like layers), and
  vLLM leaves the prefix cache off for it by default. This also means that every turn of an agent
  that runs on the local model reads its whole context again.

## Findings

1. **150,000 characters is safe with the current timeouts,** also in the worst case (new text at
   each call). This is 28% more than 117,000.
2. **For real agent traffic C2 is much faster than the worst case,** thanks to the prefix cache of
   the decision model. The limits that stay are Presidio (no cache, one worker) and the chat fallback
   of C2 on Qwen (no prefix cache).
3. **The NER on log text is still the main limit.** As on 2026-10-01, benign agent contexts under
   the cap stay LOCAL because Presidio reads pod names, ids and hostnames as personal data
   (validation K1, K3 and K8 on 2026-10-02). A higher cap does not change that.

## Not measured

- The cap of 150,000 characters end to end (validation K8-K10 after the gitops change).
- Several agents at once (Presidio queues behind its worker).
- The prefix cache of `systemone` under load from several agents, and why turn 8 is slower.
- Whether vLLM 0.24 can run Qwen3.8 with the prefix cache on.

## Reproduce

```bash
# From a LiteLLM pod (copy perf/c2_perf.py, perf/agent_growth.py and litellm/privacy_scoring.py to /tmp/perf)
python3 c2_perf.py --scenarios context --backends systemone-s1,chat \
  --decision-url http://dgemma-decision-predictor.local-models.svc.cluster.local/v1 \
  --chat-url http://qwen38-local-predictor.local-models.svc.cluster.local/v1 --chat-model qwen38-local \
  --presidio-url http://presidio-analyzer.maas-routing.svc.cluster.local:3000/analyze \
  --presidio-stop-after 15 --timeout 180 --context-sizes 32000,34000,40000,45000,54000,63000

python3 agent_growth.py \
  --decision-url http://dgemma-decision-predictor.local-models.svc.cluster.local/v1 \
  --chat-url http://qwen38-local-predictor.local-models.svc.cluster.local/v1 --chat-model qwen38-local
```
