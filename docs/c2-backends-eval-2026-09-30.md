# C2 backends: chat (Qwen3.8) and systemone (DiffusionGemma), 2026-09-30

This page compares the two backends of the C2 classifier of the privacy gate (router v0.7.0): `chat`, a
JSON verdict from the local Qwen3.8 model, and `systemone`, three yes/no questions to the decision model
(DiffusionGemma 26B-A4B FP8-dynamic, vLLM structured-read mode, example decision server). It has the
numbers for the choices that are still open: the backend for the demo, and `classifier.samples`.

> **Support status:** the decision model runs on an unsupported preview image (prototypes and proofs of
> concept). The example decision server is planned as Developer Preview in Red Hat AI Inference Server
> 3.6 GA and as Technology Preview in 3.7 EA1. LiteLLM and Presidio are community software, not
> supported by Red Hat.

## Setup

- Cluster ocp.5bw8q (OCP 4.22.15, RHOAI 3.5.1). Qwen3.8-27B-NVFP4 on one RTX PRO 6000 (g7e.2xlarge),
  thinking off. DiffusionGemma on one NVIDIA L40S (g6e.2xlarge), canvas 64.
- Router code at commit a726828 (branch `c2-systemone`), policies of `tests/policy/`, default threshold
  0.50, `CLASSIFIER_ENABLED=1 CLASSIFIER_GRAY_LOW=0` (as the demo: C2 runs on every prompt below the
  threshold).
- `eval/run_eval.py` on the three sets of `eval/`, from a workstation through `oc port-forward` to
  Presidio, Qwen and the decision server.
- `systemone` ran twice: `samples` not set (the server default, 4 noise draws) and `samples: 1`.

## Quality

| Set | Backend | Correct | Leaks (FN) | False positives | Classifier fired |
|---|---|---|---|---|---|
| english (332) | chat | 315 | 14 | 3 | 87 |
| | systemone, samples 4 | 314 | 15 | 3 | 86 |
| | systemone, samples 1 | **316** | **13** | 3 | 88 |
| italian (332) | chat | **324** | **3** | 5 | 76 |
| | systemone, samples 4 | 323 | 4 | 5 | 75 |
| | systemone, samples 1 | 323 | 4 | 5 | 75 |
| chain-demo (126) | all three | 126 | 0 | 0 | 16 |

Baseline without C2 (`eval/baseline.json`): english 101 leaks, italian 77, chain-demo 16. **No run has a
new leak** against the baseline, and no run has a leak in the `person` category.

The backends agree on most prompts: in english both fire on 81 cases, in italian on 74. The cases that
differ (english, samples 4):

- `systemone` routes LOCAL where `chat` let them go: `implicit-12`, `implicit-14`, `finance-06`,
  `finance-10`, `health-14`.
- `chat` routes LOCAL where `systemone` let them go: `legal-05`, `legal-06`, `legal-11`, `legal-14`,
  `health-13`, `finance-05`. Most are legal matters; the `personal_sensitive` question lists legal
  data, but the probability stays below 0.5 on these.
- italian: `systemone` misses `legal-14` and flags `pii-weak-04`; `chat` flags `pii-inv-06`.

## Latency

The per-case times of the eval include Presidio and the port-forward (about 2 s per case for every
backend), so they are **not** the latency of the classifier. The classifier alone was measured inside
the cluster, from the decision model pod, with the requests the router sends, on 12 prompts of
`validation` (R2, R3, R6b, R6c, D1-D8), median of 3 calls each:

| Backend | Median of the 12 medians | Range |
|---|---|---|
| systemone, samples 1 | 131 ms | 130-134 ms |
| chat on Qwen3.8 (idle GPU, thinking off) | 203 ms | 200-233 ms |
| systemone, samples 4 | 339 ms | 130-343 ms |

Other measurements on the L40S: about 570 ms for 3,800 tokens and 5.8 s for 40,000 tokens (prefill);
the first call after the start takes about 30 s (warm-up). The Qwen number is for an idle GPU: in the
demo the agents also use Qwen, and C2 then waits behind their requests. The decision model has its own
GPU.

Verdicts on the demo prompts: both backends agree on R2, R3, R6b, R6c, D1-D6 and D8. On D7 ("Mario
Rossi, an Italian engineer who lives in Milan", expected LOCAL) `systemone` answers 0.88 (0.96 with
samples 1) and `chat` answers not sensitive; the rules already route D7 LOCAL.

## What this suggests (to be decided)

- Quality: the two backends are close. `systemone` with `samples: 1` has the fewest leaks in english and
  the same result as `samples` 4 in italian.
- Latency: `samples: 1` is 2.6 times faster than the default and faster than Qwen on an idle GPU.
- Open proposals, not decided: `classifier.samples: 1` for the demo; a question wording that catches the
  `legal-*` cases; the probability as the signal weight (today fixed at 0.80).
