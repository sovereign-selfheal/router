# Privacy gate and C2 with agents' large contexts, 2026-10-01

The self-heal agents send long requests: a system prompt, the incident, rounds of tool calls and their
output (logs, events, tickets), and a question. This page measures how the privacy gate and the C2
classifier behave when the context grows from about 8,000 to 255,000 tokens.

> **Support status:** the decision model (`systemone`) runs on an unsupported preview image. LiteLLM
> and Presidio are community software, not supported by Red Hat.

## What the router does with a large request

The privacy gate scores the **whole payload** (`_whole_payload` in `litellm/policy_hook_chain.py`):
the system prompt, every turn, the tool output and the tool call arguments. A privacy control must not
look at the last message only. So the whole context of an agent goes through:

1. the rules (regex, lexicons): fast;
2. Presidio NER, timeout 3 s (`ner.timeout_seconds`). On an error or a timeout: fail-closed, LOCAL;
3. C2, only when the rules score is below the threshold. Timeout 8 s (`classifier.timeout_seconds`),
   then the chat fallback, 8 s more, then fail-closed.

## Setup

- Cluster ocp.5bw8q, router v0.7.0, `classifier.samples: 1`, decision model on (L40S), Qwen3.8 on an
  RTX PRO 6000 (maximum context 131,072 tokens), Presidio 2 replicas (gunicorn, 1 worker, 4 threads).
- Text like the agents' tool output: Kubernetes log lines, events and ticket lines, with a sensitive
  sentence (a colleague's divorce and sick leave) in the middle where the case says so.
- Three levels:
  1. components alone, from a LiteLLM pod (the path of the router):
     [`perf/c2_perf.py --scenarios context`](../perf/c2_perf.py), 3 calls per size;
  2. end to end through the public route, streamed, as an agent: validation group `context`
     (K1-K7, `harness/load.py` of the validation repo);
  3. 4 agents in parallel, 30,000 tokens each.
- Raw data: [`perf/results/2026-10-01-ocp.5bw8q-context.jsonl`](../perf/results/2026-10-01-ocp.5bw8q-context.jsonl).
- Load only. Presidio stayed Ready with no restart during the whole test (116 checks, every 10 s).
  Presidio was not sent texts above 285,000 characters, to protect its single worker.

## 1. Components alone (one call at a time, p50)

| Context (tokens) | Presidio /analyze (timeout 3 s) | C2 systemone (timeout 8 s) | C2 chat on Qwen3.8 (timeout 8 s) | Sensitive sentence found |
|---|---|---|---|---|
| about 8,000 (18,000 characters) | 0.66 s | 0.69 s | 0.79 s | yes, both |
| about 31,000 (71,000 characters) | 2.57 s | 3.78 s | 3.28 s | yes, both |
| about 62,000 (143,000 characters) | **5.0 s, over** | **10.6 s, over** | **8.1 s, over** | yes, both |
| about 125,000 (285,000 characters) | **10.5 s, over** | 34.9 s | 23.4 s | yes, both |
| about 198,000 | not sent | 78.4 s | **rejected** (over 131,072) | systemone: yes |
| about 255,000 | not sent | over 120 s (client timeout) | **rejected** | no answer |

- Presidio passes its 3 s timeout at about **35,000 tokens**.
- C2 passes its 8 s timeout at about **45,000-50,000 tokens**, on both backends. On long prompts the
  decision model (L40S) is slower than Qwen3.8 (RTX PRO 6000).
- Quality is not the problem: both backends found the sensitive sentence at every size they answered.

## 2. End to end, as an agent (validation group `context`)

| Id | Context | Prompt tokens | First chunk | Routed | Why |
|---|---|---|---|---|---|
| K1 | benign | 8,577 | 2.2 s | LOCAL | Presidio entities in the logs: PERSON, LOCATION |
| K2 | sensitive sentence | 8,672 | 2.2 s | LOCAL | the same entities (C2 not needed) |
| K3 | benign | 30,892 | 6.6 s | LOCAL | MEDICAL_LICENSE, NRP, PERSON, LOCATION in the logs |
| K4 | sensitive sentence | 30,988 | 6.7 s | LOCAL | the same entities |
| K5 | benign | 61,615 | 12.3 s | LOCAL | `error:ReadTimeout`: Presidio timeout, fail-closed (WARN) |
| K6 | sensitive sentence | 61,711 | 12.9 s | LOCAL | Presidio timeout, fail-closed |
| K7 | benign | 101,796 | 22.1 s | LOCAL | Presidio timeout, fail-closed (WARN) |

Result: 5 PASS, 2 WARN, no FAIL. **No sensitive context left the cluster.** But no benign context
could go to SOTA either.

## 3. Four agents in parallel, 30,000 tokens each

All four answered 200 (first chunk 7.4 s, 10.9 s, 14.3 s, 16.5 s; total about 20 s). Three of the four
stayed LOCAL with `error:ReadTimeout`: alone a 30,000-token text takes 2.6 s in Presidio, but four at
once queue behind each other and pass 3 s.

## Findings

1. **Presidio is the first limit, not C2.** Above about 35,000 tokens (or less with parallel agents)
   Presidio times out, the gate fails closed, and the request stays LOCAL before C2 runs.
2. **Technical text gives Presidio false positives.** Pod, node and image names, ids and hostnames in
   logs read as PERSON, LOCATION, NRP and MEDICAL_LICENSE. Even a benign 8,000-token context passes the
   threshold with the rules alone, so C2 is never asked.
3. **C2 has a limit of about 45,000-50,000 tokens** with an 8 s timeout, on both backends. Beyond that,
   C2 and its fallback time out (up to 16 s) and the gate fails closed.
4. **Above 131,072 tokens Qwen3.8 cannot serve the request at all**, LOCAL or as the C2 fallback.
5. **The time to the first token grows with the context**: 2 s at 8,000 tokens, 22 s at 100,000.
6. **Safety holds**: every failure ends LOCAL (fail-closed). The cost is that agents' requests
   practically never go to SOTA, and C2 (with either backend) has almost no role for them.

## Options (to be decided, not applied)

- **What the gate reads for agents.** For example, score the system prompt, the user turns and the tool
  call arguments fully, and the tool output in a bounded way (a size cap, or chunks scored in
  parallel). This changes the privacy semantics: it needs a design decision.
- **Presidio for long texts.** Chunks in parallel, more replicas or workers, or a timeout that grows with
  the size.
- **Technical-text false positives.** Tune `ner.entity_weights` and `ner.context_entities` for log
  text, measured on an eval set of agent contexts.
- **C2 on a bounded window.** For example the first and last parts of the context, so it stays under
  its timeout.
- **A policy for agents.** For example the agents' tier always LOCAL by design. This is simple and
  sovereign, but SOTA is never used for agents.

## Not measured

- Real OGX payloads: the agents are not deployed on this cluster yet.
- Italian contexts, and contexts with a lot of personal data (customer tickets).
- Presidio above 285,000 characters, and Presidio with more workers.
