# router: LiteLLM hook code and image

Code of the smart router of the demo *The Sovereign, Self-Healing Platform*. For each request, a LiteLLM
hook decides whether the **local model** or the external **SOTA model** answers. Read
[`AGENTS.md`](AGENTS.md) before changing anything.

> **Support status:** LiteLLM is community software, operated by the customer, **not** supported by
> Red Hat. The privacy gate also calls Presidio (community, see the `presidio` repo). Routing between a
> local and an external model is Tech Preview in Red Hat OpenShift AI 3.5. The `systemone` backend of the
> C2 classifier (since v0.7.0) calls the example decision server of vLLM, which runs on an **unsupported
> preview image** (DiffusionGemma, vLLM structured-read mode). It is planned as Developer Preview in Red
> Hat AI Inference Server 3.6 GA and as Technology Preview in 3.7 EA1; its API may change.

This repo delivers two things with the **same version** (git tag `vX.Y.Z`):

| Item | Where it runs | How it reaches the cluster |
|---|---|---|
| Hook code: `litellm/policy_hook_chain.py`, `litellm/privacy_scoring.py`, `litellm/namespace_policy.py` (since v0.11.0) | LiteLLM pod, `/app/litellm` | The `gitops` repo copies the files at the tag into a ConfigMap (`scripts/sync-router-code.sh` there) |
| Image `quay.io/sovereign-selfheal/router:vX.Y.Z`: LiteLLM v1.102.0 + fastText + `lid.176.ftz` | LiteLLM pod | The `gitops` repo pins it by digest |

The policies (`chain.yaml`, `privacy-plus.yaml`) are **not** here: the `gitops` repo is their source of
truth. `tests/policy/` holds a copy for the tests and the evaluation.

## How the hook decides

![Animated diagram of the routing chain: a request passes the namespace gate, the efficiency gate, the
privacy gate and the tiering; the first step that says LOCAL sends it to the local GPU model, otherwise
the external SOTA model answers](docs/img/routing-chain.svg)

The animation shows five example requests, one after the other. Its source is
`scripts/routing_chain_svg.py`: change it when the chain changes, then run
`python3 scripts/routing_chain_svg.py docs/img/routing-chain.svg`.

`policy_hook_chain.py` runs two gates in order. The first gate that says LOCAL wins. Since v0.11.0 an
optional **namespace policy** runs before them (off by default; see "Namespace policy"): a request
about a restricted namespace stays local and no gate runs.

1. **Efficiency**: short and simple questions stay local; long questions, or questions with a
   complexity keyword, can go to SOTA. A request larger than the SOTA size cap stays local (since
   v0.8.0, off by default; see "Large requests").
2. **Privacy** (`privacy_scoring.py`): rules (regex with validators, lexicons), Presidio NER and an
   optional LLM classifier give a score. At or above the threshold of the team, the request stays local.

Then the **tiering** of the team can only move the decision to LOCAL. Any unexpected error routes LOCAL
(fail-closed). Every decision is logged as one line, `[policy-router] {...}`.

## Policy keys by version

The code reads its settings from the gitops policies. A new key always has a default that keeps the
previous behaviour, so an old policy works with a new release.

| Key (in `privacy-plus.yaml`) | Since | Default | Meaning |
|---|---|---|---|
| `ner.person_min_words` | v0.4.0 | `1` | A `PERSON` entity counts only with at least this many words. `2` ignores single words that Presidio reads as names ("Kafka", "Paxos", "Spiega"); a full name like "Mario Rossi" still counts |
| `classifier.backend` | v0.7.0 | `chat` | Backend of the C2 classifier: `chat` (one JSON verdict from a chat model, as before) or `systemone` (a decision server, see "C2 backends"). Env override: `CLASSIFIER_BACKEND` |
| `classifier.decision_threshold` | v0.7.0 | `0.5` | Backend `systemone`: a positive question at or above this probability adds the C2 signal |
| `classifier.samples` | v0.7.0 | none | Backend `systemone`: noise draws per question; none = the default of the decision server (4) |
| `classifier.fallback_base_url_env` / `fallback_model_env` | v0.7.0 | `CLASSIFIER_FALLBACK_BASE_URL` / `CLASSIFIER_FALLBACK_MODEL` | Backend `systemone`: env vars of the chat fallback |
| `ner.context_entities` | v0.4.0 | `[]` | These entity types (for example `NRP`, `LOCATION`) count only when the text also has an identifier: structured personal data (card, IBAN, email, phone...) or another entity that counts. "European banks in Italy" names nobody |
| `ner.timeout_per_1k_chars` / `classifier.timeout_per_1k_chars` | v0.8.0 | `0` | Seconds of timeout per 1,000 characters of text, for Presidio and for each C2 call. `0` = the fixed `timeout_seconds`, as before. See "Large requests" |
| `ner.timeout_max_seconds` / `classifier.timeout_max_seconds` | v0.8.0 | `timeout_seconds` | Upper bound of the timeout that grows with the text |
| `classifier.systemone.questions` | v0.9.0 | the built-in questions | Backend `systemone`: the yes/no questions, `{id: {instructions: <text>, ignored: <bool>}}`. Positive questions add the C2 signal; `ignored: true` questions only show in the log. An invalid set (no positive question, an empty or missing `instructions`) is logged and the built-in questions are used. Keep the ids short: the answer template must fit the canvas of the decision server (64 tokens: about ten questions with one-token ids) |
| `classifier.systemone.show_all` | v0.9.0 | `false` | Backend `systemone`: every positive question below the threshold also shows in the log with weight 0, for example `systemone/health@0.03(shown):0.00`. For evaluations |

Keys in `chain.yaml`: `efficiency.sota_max_prompt_chars` (since v0.8.0, default `0` = no cap), the
SOTA size cap of "Large requests"; the block `namespace_policy` (since v0.11.0, default off), see
"Namespace policy".

The key `classifier.chat_template_kwargs` (since v0.6.0, default: none) is a mapping of chat template
arguments sent with the call of the C2 classifier. The env var `CLASSIFIER_CHAT_TEMPLATE_KWARGS` (a JSON
object) overrides it. The gitops repo sets `{"enable_thinking": false}` when the classifier is the local
Qwen3 model, which reasons by default: with reasoning, the classifier would spend its token budget and
return no JSON verdict. Do not set it for a provider that does not accept this parameter. An invalid
value (not a mapping, or an env var that is not a JSON object) is ignored with a `[policy-router]` log line.

The env var `NER_ENABLED` (since v0.10.0) overrides `ner.enabled` of the policy: `1`, `true`, `yes` or
`on` turn C1 (Presidio NER) on, any other value turns it off. Unset or empty, the policy decides. With
C1 off, Presidio gets no calls and the log has no `ner` signals; the log line still shows
`ner_timeout_s`. The gitops repo turns C1 off when the decision model answers C2 (see its README).
Quality and gate time with C1 on and off: [`docs/c1-off-eval-2026-10-05.md`](docs/c1-off-eval-2026-10-05.md).

Ignored entities stay in the log with weight 0, for example `NRP@0.85(no id):0.00` or
`PERSON@0.85(<2 words):0.00`, so the log shows what the engine saw and why it did not count it.

## Namespace policy (since v0.11.0)

The same agent can investigate a sensitive application and an ordinary one. A platform team marks the
namespace of a sensitive application with a label, and every request about that namespace stays on the
local model:

```bash
oc label namespace payments sovereign-selfheal.io/data-class=restricted --overwrite
```

**Only `restricted` changes the routing.** A request about a restricted namespace goes to `local-fast`
before the gates (`decided_by: namespace`): no detector runs, so the decision is also faster. No label,
`public` or any other value keeps the normal routing (efficiency and privacy gates). `public` only says
that someone classified the namespace. An unknown value, for example a typo, keeps the normal routing
and shows in the metric `router_namespace_labels{state="unknown"}`.

**How the router knows the namespaces of a request.** Two ways, each with its own switch. One
restricted namespace from either way is enough.

| Switch | Way | Notes |
|---|---|---|
| `hint` | The agent sends the names in the request body: `"selfheal_namespaces": ["payments"]` | Precise. Contract for agents: [`docs/namespace-policy.md`](docs/namespace-policy.md) |
| `scan` | The router finds names in the text of the request: PromQL label matchers (`namespace="payments"`, every alternative of `namespace=~"a\|b"`, wildcards like `"pay.*"` against the labelled names), JSON and YAML `namespace` keys, the alert labels of ogx-alert-translator (``- `namespace`: payments``), `payments.svc`, `-n payments` (also in a JSON argv array), `/namespaces/payments`; also inside JSON-encoded tool call arguments (v0.11.1) | Needs no change in the agent. A name that never appears in the text is not seen. It also catches a tool that reads a restricted namespace during an investigation of another one. About 8 ms for 400,000 characters (v0.11.0; the v0.11.1 forms add a few ms) |

The hint field is removed from **every** request, also when the switch is off, so LiteLLM never sends
it to a model.

**Labels.** The router lists the namespaces that have the label key, with the ServiceAccount of the pod
(`get`/`list`/`watch` on namespaces, given by the gitops repo). The first read happens when the hook
loads, then a thread reads again every `refresh_s` seconds. A failed read keeps the last list. When the
list was never read, nothing counts as restricted: the normal routing applies, the log line shows
`ns_labels_loaded: False` and `router_namespace_labels_loaded` is 0.

| Key (in `chain.yaml`, block `namespace_policy`) | Default | Meaning |
|---|---|---|
| `scan` | `false` | Find namespace names in the request text. Env override: `NAMESPACE_SCAN_ENABLED` |
| `hint` | `false` | Read the names that the agent sends. Env override: `NAMESPACE_HINT_ENABLED` |
| `label` | `sovereign-selfheal.io/data-class` | Label key on the namespaces |
| `hint_field` | `selfheal_namespaces` | Field of the request body with the names |
| `refresh_s` | `5` | Seconds between two reads of the labels (minimum 1) |

The env overrides accept `1`, `true`, `yes` or `on` (on) and any other value (off); unset or empty, the
policy decides. With both switches off the hook makes no Kubernetes API call and starts no thread.

With the policy on, the log line has four more keys: `ns_restricted` (the restricted names found),
`ns_source` (`hint`, `scan`, `hint+scan` or `none`), `namespaces` (the hint names and the labelled names
that the scan found) and `ns_labels_loaded`. The trace has one more span, `gate.namespace` (verdict
`local` or `pass`).

## Large requests (since v0.8.0)

The self-heal agents send the whole conversation each turn: system prompt, tool definitions, tool calls
and their output. These requests can be larger than the SOTA model accepts, and Presidio and C2 need more
time for them. Measurements: [`docs/c2-large-context-2026-10-01.md`](docs/c2-large-context-2026-10-01.md).
The values below, and the tests behind them: [`docs/sota-size-cap-2026-10-01.md`](docs/sota-size-cap-2026-10-01.md).
Both features are off by default.

**SOTA size cap** (`efficiency.sota_max_prompt_chars` in `chain.yaml`). The efficiency gate measures the
whole request in characters: every message, the tool call arguments and the tool definitions. Above the
cap the request stays LOCAL with the reason `efficiency: SOTA context limit (<size> > <cap> chars)`, and
no detector runs (no Presidio call, no C2 call). An error while measuring also routes LOCAL. The cap is
in characters, not in tokens: the router has no tokenizer, and the SOTA model would count with its own.
Choose it as the smaller of two limits: `(SOTA context window - max_tokens of the agents - margin) x
characters per token`, and the largest text that Presidio and C2 read within their timeouts. Log text
has about 2.3 characters per token for Qwen and about 1.6 for Gemini, English prose about 4. With a
65,536-token window the first limit gave 117,000 characters
([2026-10-01](docs/sota-size-cap-2026-10-01.md)). With gemini-2.5-pro the detectors set the limit:
150,000 characters ([2026-10-02](docs/sota-size-cap-2026-10-02.md)).

**Timeouts that grow with the text** (`ner.*` and `classifier.*` in `privacy-plus.yaml`). The timeout of
one call is

```
min(max(timeout_seconds, timeout_per_1k_chars x characters / 1000), max(timeout_seconds, timeout_max_seconds))
```

where the characters are those of the text sent to the detector. With `timeout_per_1k_chars: 0` the
timeout is `timeout_seconds`, as before. A timeout still fails closed. Values from the measurements of
2026-10-01, with a margin of about 1.4-1.5x: Presidio `0.052` s per 1,000 characters, max `10`;
C2 `0.11`, max `15`. The bound is per call: Presidio is called twice (`en` and `it`) when the language
is uncertain, and C2 with `systemone` calls the chat fallback after a failure. The worst case of the
gate with these values is 2 x 10 + 2 x 15 = 50 s.

**Log line.** The decision has four more keys (additive, the previous keys do not change):
`prompt_chars` (size of the whole request, `None` if it could not be measured), `sota_cap` (`None` =
off), and, when the privacy gate ran, `ner_timeout_s` and `c2_timeout_s` (the effective timeouts).

## C2 backends (since v0.7.0)

The C2 classifier runs only in the gray zone of the privacy score, and it can only add a signal.

| Backend | Call | Signal when sensitive |
|---|---|---|
| `chat` (default) | `POST $CLASSIFIER_BASE_URL/chat/completions`: one JSON verdict `{"sensitive", "confidence"}` | `llm@0.90` |
| `systemone` | `POST $CLASSIFIER_BASE_URL/systemone` (the example decision server of vLLM): yes/no questions with the text as state, one probability each from one forward pass | `systemone/llm@0.93(legal)` |

The built-in `systemone` questions keep the criteria of the chat prompt: `personal_sensitive` and
`credentials` add the signal (weight 0.80, as the chat backend) when the higher of the two is at or above
`classifier.decision_threshold`. Since v0.9.0 the policy can replace them
(`classifier.systemone.questions`), and the label names the positive question with the highest
probability: `systemone/llm@0.93(legal)`. The label still contains `llm@<p>`, as before. See
[`docs/c2-questions-eval-2026-10-03.md`](docs/c2-questions-eval-2026-10-03.md) for an evaluation of
split questions. `business_confidential` never adds weight; it stays in the log with
weight 0, for example `systemone/business_confidential@0.99(ignored):0.00`, so the log shows why a
confidential business text may still go to SOTA.

**Fallback chain.** With `systemone`, an error, a timeout (`timeout_seconds`, 8 s by default) or a reply
without a probability for every question does not fail closed at once. The hook asks the chat backend at
`CLASSIFIER_FALLBACK_BASE_URL` / `CLASSIFIER_FALLBACK_MODEL` (the local Qwen model in the demo) with the
chat prompt; its signal is `fallback/llm@0.90`. Only when the fallback fails too, or is not set, the
signal is `error` (fail-closed). The worst case is two timeouts, 16 s. The span `gate.privacy` gets the
attributes `classifier.backend` and `classifier.fallback` (boolean). The fallback matters in the demo:
with `grayLow: "0"`, a fail-closed C2 would send every request LOCAL.

The evaluation of both backends (quality and latency) is in
[`docs/c2-backends-eval-2026-09-30.md`](docs/c2-backends-eval-2026-09-30.md). Parallel calls, prompt
size and C2 while the local model is busy are in
[`docs/c2-backends-performance-2026-10-01.md`](docs/c2-backends-performance-2026-10-01.md); the test
script is [`perf/c2_perf.py`](perf/c2_perf.py). The privacy gate and C2 with agents' large contexts (8,000 to 255,000
tokens) are in [`docs/c2-large-context-2026-10-01.md`](docs/c2-large-context-2026-10-01.md). The vLLM prefix cache of the
local Qwen model (before and after it was turned on) is in
[`docs/qwen-prefix-cache-2026-10-02.md`](docs/qwen-prefix-cache-2026-10-02.md).

Measured on 2026-09-30 on one NVIDIA L40S (the decision server with three questions, a new text each
call): about 350 ms for a short prompt (134 ms with `samples: 1`), 570 ms for 3,800 tokens, 5.8 s for
40,000 tokens. The first call after the model starts takes about 30 s.

## Traces and metrics (since v0.5.0)

The hook shows each decision as a trace and as metrics. **Instrumentation is never on the decision
path**: every tracing or metrics error is logged (`[policy-router] tracing error (ignored)`) and
ignored, and the decision is the same with or without OpenTelemetry and `prometheus_client`. Both
packages are in the LiteLLM image (opentelemetry 1.28.0, prometheus_client 0.20.0, extra
`proxy-runtime` of LiteLLM 1.102.0); without them the hook works as before.

**Traces.** With the `otel` callback of LiteLLM on, LiteLLM creates one span per proxy request and
passes it to the hook (`metadata.litellm_parent_otel_span`). The hook adds:

```
<proxy request span>                      (LiteLLM)
├── router.chain                          route.requested, route.target, route.decided_by,
│   │                                     route.team, route.reason
│   ├── gate.namespace                    gate.verdict (local|pass), gate.reason (v0.11.0, policy on)
│   ├── gate.efficiency                   gate.verdict, gate.reason
│   └── gate.privacy                      gate.verdict, gate.reason, privacy.score,
│       │                                 privacy.threshold, privacy.team_key, privacy.signals
│       └── presidio.analyze (per lang)   presidio.language, presidio.entities_found
└── litellm_request                       (LiteLLM, with USE_OTEL_LITELLM_REQUEST_SPAN=true):
                                          the model call; hidden_params has the api_base
```

A gate that is not evaluated (short-circuit) has no span. An error (for example Presidio down) is
recorded on its span. Without a parent span, `router.chain` is a root span; without the `otel` callback
the spans are no-ops. The decision gets one more key, `trace_id` (32 hex characters), only when a
trace exists: the log line keeps all its previous keys.

**Metrics.** The hook serves the default `prometheus_client` registry on `ROUTER_METRICS_PORT`:
its own metrics, and the ones of LiteLLM's `prometheus` callback when that is on.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `router_requests_total` | counter | `routed_to`, `decided_by`, `team` | One per decision. `decided_by`: `namespace` (v0.11.0), `efficiency`, `privacy`, `tiering`, `all-sota`, `fail-closed`. `team` is a team named in the policies, `none` or `other` |
| `router_privacy_score` | histogram | `team` (threshold key) | Privacy score of the requests that reached the privacy gate |
| `router_sota_budget_used_tokens` | gauge | | SOTA tokens counted by the budget of the efficiency gate, **per pod** |
| `router_namespace_decisions_total` | counter | `target_namespace`, `routed_to`, `source` | v0.11.0, namespace policy on: one per decision and restricted namespace. `target_namespace` is a restricted namespace or `none` (a bounded set: the platform team sets the labels; not `namespace`, which Prometheus sets to the scrape namespace) |
| `router_namespace_labels_loaded` | gauge | | v0.11.0: 1 when this pod read the namespace labels at least once |
| `router_namespace_labels` | gauge | `state` | v0.11.0: labelled namespaces, `restricted`, `public` or `unknown` (another value) |

Tokens and fallbacks per model come from LiteLLM's `prometheus` callback
(`litellm_total_tokens_metric_total{requested_model=...}`,
`litellm_deployment_successful_fallbacks_total{requested_model="sota-smart",fallback_model="local-fast"}`).

| Env var | Read by | Default | Meaning |
|---|---|---|---|
| `ROUTER_METRICS_PORT` | hook | `9091` | Port of the metrics server; `0` = off. Started once per process; a bind error is logged and the router works without metrics. Off when `PROMETHEUS_MULTIPROC_DIR` is set |
| `OTEL_EXPORTER`, `OTEL_ENDPOINT`, `OTEL_SERVICE_NAME` | LiteLLM `otel` callback | | Where the traces go (the gitops repo sets the in-cluster collector) |
| `USE_OTEL_LITELLM_REQUEST_SPAN` | LiteLLM | `false` | `true`: a child span for the model call |

## Develop and test

The Python tools are managed with [uv](https://docs.astral.sh/uv/), never pip.

```bash
uv sync                              # creates .venv from uv.lock
uv run pytest                        # unit tests: no network, no LiteLLM, no Presidio
uv run ruff check .
```

The unit tests use a fake Presidio and a fake LiteLLM module (`tests/conftest.py`).

## Evaluation

`eval/` has three sets of labelled prompts (all personal data is synthetic): `privacy-plus-english.yaml`,
`privacy-plus-italian.yaml` (332 cases each) and `chain-demo.yaml` (126 cases, gate order).
`eval/run_eval.py` scores them with the real engine. `eval/check_baseline.py` runs every set and fails on
a **new leak**: a case that must stay LOCAL, goes to SOTA, and is not in `eval/baseline.json`.

The sets contain cases that fail on purpose (for example the `implicit` ones, which need the LLM
classifier). The baseline records them, so only a change for the worse fails.

```bash
scripts/fetch-lid-model.sh           # fastText model into .cache/, sha256 checked
podman run -d --rm -p 3000:3000 quay.io/sovereign-selfheal/presidio:<version>
uv run eval/check_baseline.py        # compare with eval/baseline.json
uv run eval/check_baseline.py --update   # accept the new results (explain why in the PR)
```

The classifier stays off in the evaluation, and the harness uses the default threshold of the policy.
To compare the C2 backends, run `eval/run_eval.py` against a cluster (port-forward Presidio and the
classifier) with `CLASSIFIER_ENABLED=1 CLASSIFIER_GRAY_LOW=0`, then `CLASSIFIER_BACKEND=chat` or
`systemone` and the `CLASSIFIER_*` URLs; `CLASSIFIER_SAMPLES` sets `classifier.samples`. A row counts as
`classifier_fired` when a classifier signal has weight above 0.

## Release

Quay builds the image. A build trigger on the Quay repository follows the git tags of this repo:

1. Merge the change on `main`. The CI must be green.
2. Create and push a tag `vX.Y.Z` (`git tag v0.4.0 && git push origin v0.4.0`).
3. Quay builds `Containerfile` and publishes `quay.io/sovereign-selfheal/router:vX.Y.Z`.
4. In the `gitops` repo, run `scripts/sync-router-code.sh sync vX.Y.Z`. It copies the hook code at the
   tag and sets `routerVersion`. Then pin the image digest of the same tag in
   `components/litellm-router/values.yaml` and open a PR.

Never move or reuse a tag. To read the digest of a tag:

```bash
curl -fsS "https://quay.io/api/v1/repository/sovereign-selfheal/router/tag/?specificTag=vX.Y.Z" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["tags"][0]["manifest_digest"])'
```

Quay setup (once): public repository `sovereign-selfheal/router`, build trigger on the GitHub repo
`sovereign-selfheal/router`, only for refs that match `tags/v.*`, Dockerfile `/Containerfile`, context
`/`, image tag = git tag name (`${parsed_ref.tag}`), no `latest`.

## Check your changes (also run by CI)

```bash
uv sync --locked
uv run ruff check .
uv run yamllint .
shellcheck scripts/*.sh
hadolint Containerfile
uv run pytest
uv run eval/check_baseline.py        # needs Presidio on localhost:3000, see "Evaluation"
podman build -f Containerfile -t localhost/router:dev .
```

## License

Apache License 2.0, see [LICENSE](LICENSE). The hook code and the evaluation sets come from the old
repository `rhocpai-mvp-routing` (BSD 3-Clause, same author).
