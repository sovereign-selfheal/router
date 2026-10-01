# C2 backends: performance, 2026-10-01

This page measures the latency and the throughput of the two backends of the C2 classifier: `systemone`
(the decision model, DiffusionGemma 26B-A4B FP8-dynamic behind the example decision server) and `chat`
(the local Qwen3.8 model, a JSON verdict). The quality of the two backends is in
[`c2-backends-eval-2026-09-30.md`](c2-backends-eval-2026-09-30.md).

> **Support status:** the decision model runs on an unsupported preview image (prototypes and proofs of
> concept). The example decision server is planned as Developer Preview in Red Hat AI Inference Server
> 3.6 GA and as Technology Preview in 3.7 EA1.

## Setup

- Cluster ocp.5bw8q (OCP 4.22.15, RHOAI 3.5.1), router v0.7.0 with the decision model on.
  - Decision model: one NVIDIA L40S (g6e.2xlarge), vLLM 0.29.1.dev1248 (preview image), flags of
    `gitops/bootstrap/values.yaml` (`--max-num-seqs=4`, `--enforce-eager`, canvas 64).
  - Qwen3.8-27B-NVFP4: one RTX PRO 6000 Blackwell (g7e.2xlarge), RHOAI vLLM 0.24.0, MTP.
- The client is [`perf/c2_perf.py`](../perf/c2_perf.py), in a temporary pod in the model namespace
  (`ubi9/python-312`), so no port-forward and no router are in the path. It sends the requests of the
  router: the three yes/no questions of `systemone`, or the chat prompt with thinking off and
  `max_tokens` 200.
- Every prompt is unique (a random prefix), so the prefix cache of vLLM never helps.
- The decision server was tested twice:
  - `systemone-auto`: no `samples` key, the server default `"auto"`: 1 to 4 noise draws per question,
    more when the answer is uncertain;
  - `systemone-s1`: `"samples": 1`.
- Token counts are the ones the servers report. They include the questions (`systemone`) or the
  system prompt (`chat`), so the same text gives slightly different counts.
- Raw data: [`perf/results/2026-10-01-ocp.5bw8q.jsonl`](../perf/results/2026-10-01-ocp.5bw8q.jsonl).
  All 48 measurements had 0 errors.
- Only load, nothing was restarted or changed. Other users shared the cluster during the test.

## 1. Parallel C2 calls

Short prompts (about 600 tokens with the questions), 20 to 128 calls per level, all clients start
together. Latency in ms (p50 / p95), throughput in calls per second.

| Parallel calls | systemone-s1 | systemone-auto | chat (Qwen3.8) |
|---|---|---|---|
| 1 | 142 / 147, 7.0/s | 145 / 379, 5.0/s | 242 / 274, 4.0/s |
| 2 | 217 / 219, 9.2/s | 214 / 2,068, 4.7/s | 339 / 372, 5.8/s |
| 4 | 230 / 232, 17.3/s | 460 / 2,184, 5.4/s | 428 / 551, 9.1/s |
| 8 | 427 / 510, 18.5/s | 818 / 1,449, 8.2/s | 586 / 673, 13.4/s |
| 16 | 861 / 1,058, 17.5/s | 1,033 / 2,080, 12.9/s | 936 / 1,234, 15.9/s |
| 32 | 1,668 / 1,707, 18.9/s | 2,257 / 5,145, 11.0/s | 1,670 / 2,131, 18.8/s |

- `systemone-s1` is the fastest backend at up to 4 parallel calls. It saturates at about 18-19 calls
  per second from 4 parallel calls on; above that the latency grows linearly (queueing).
- `systemone-auto` has a long tail. When the server adds noise draws, one call costs up to 4 forward
  passes, so p95 is 2-5 s under load and the throughput is lower.
- Qwen3.8 on its larger GPU reaches the same throughput as `systemone-s1` at 32 parallel calls, with
  higher latency at low load.
- `--max-num-seqs=4` (the flag of the quick start) limits the batch of the decision server. A larger
  value may raise its throughput; this is not tested.

## 2. Prompt size

One call at a time, 5 calls per size, a sensitive sentence at the end of the text. Latency in ms
(p50 / p95).

| Tokens (reported) | systemone-s1 | systemone-auto | chat (Qwen3.8) |
|---|---|---|---|
| about 650 | 143 / 149 | 363 / 369 | 242 / 253 |
| about 3,400 | 271 / 272 | 271 / 1,021 | 416 / 443 |
| about 12,600 | 1,089 / 1,090 | 1,086 / 1,394 | 1,244 / 1,275 |
| about 31,100 | 3,543 / 3,550 | 3,532 / 3,691 | 3,402 / 3,407 |

- Long prompts are bound by prefill on both GPUs: about 3.5 s at 31,000 tokens for both backends.
- The C2 timeout is 8 s (`classifier.timeout_seconds`). Both backends stay below it at 31,000 tokens.
  Longer prompts were not measured.

## 3. C2 while Qwen3.8 is busy (the agents' load)

Background: N threads ask Qwen3.8 for long answers (`max_tokens` 1500, thinking on), as the self-heal
agents do, on prompts of about 600 or 8,000 tokens. Foreground: 20 C2 calls, one at a time, per backend.
Latency in ms (p50 / p95). Idle values are from section 1, one parallel call.

| Qwen3.8 load | systemone-s1 | systemone-auto | chat (Qwen3.8) |
|---|---|---|---|
| idle | 142 / 147 | 145 / 379 | 242 / 274 |
| 4 streams, 600-token prompts | 142 / 160 | 143 / 356 | 290 / 322 |
| 8 streams, 600-token prompts | 141 / 147 | 143 / 355 | 339 / 491 |
| 16 streams, 600-token prompts | 139 / 146 | 141 / 351 | 340 / 478 |
| 32 streams, 600-token prompts | 139 / 150 | 140 / 347 | 415 / 469 |
| 8 streams, 8,000-token prompts | 141 / 209 | 143 / 365 | 315 / 797 |
| 16 streams, 8,000-token prompts | 139 / 145 | 143 / 357 | 353 / 849 |

- The decision model does not see the load of Qwen3.8: it has its own GPU.
- C2 on Qwen3.8 slows down with the load: p50 up to 1.7 times the idle value with 32 streams, and p95
  up to 3 times with long prompts. Each long prompt that Qwen3.8 reads (prefill) delays the C2 calls
  behind it.
- In the demo, every request that goes to the router while the agents work waits for C2 first. With
  the decision model this wait stays at about 140 ms.

## What this suggests (to be decided)

- `classifier.samples: 1` gives the lowest and steadiest latency, and the highest throughput of the
  decision server. With the eval of 2026-09-30 its quality is the same as or better than the default.
- `--max-num-seqs` of the decision server could be raised for more parallel C2 calls; it needs a test.
- The two backends have similar throughput. The argument for the decision model is isolation: a steady
  C2 latency while Qwen3.8 serves the agents.

## Not measured

- C2 calls and agents' calls to the decision model at the same time (the agents do not use it yet).
- The end-to-end latency added by the router (LiteLLM, Presidio, the rules) on top of C2.
- Prompts longer than about 31,000 tokens, and the C2 timeout with them.
