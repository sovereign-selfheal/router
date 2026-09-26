# router: LiteLLM hook code and image

Code of the smart router of the demo *The Sovereign, Self-Healing Platform*. For each request, a LiteLLM
hook decides whether the **local model** or the external **SOTA model** answers. Read
[`AGENTS.md`](AGENTS.md) before changing anything.

> **Support status:** LiteLLM is community software, operated by the customer, **not** supported by
> Red Hat. The privacy gate also calls Presidio (community, see the `presidio` repo). Routing between a
> local and an external model is Tech Preview in Red Hat OpenShift AI 3.5.

This repo delivers two things with the **same version** (git tag `vX.Y.Z`):

| Item | Where it runs | How it reaches the cluster |
|---|---|---|
| Hook code: `litellm/policy_hook_chain.py`, `litellm/privacy_scoring.py` | LiteLLM pod, `/app/litellm` | The `gitops` repo copies the files at the tag into a ConfigMap (`scripts/sync-router-code.sh` there) |
| Image `quay.io/sovereign-selfheal/router:vX.Y.Z`: LiteLLM v1.102.0 + fastText + `lid.176.ftz` | LiteLLM pod | The `gitops` repo pins it by digest |

The policies (`chain.yaml`, `privacy-plus.yaml`) are **not** here: the `gitops` repo is their source of
truth. `tests/policy/` holds a copy for the tests and the evaluation.

## How the hook decides

`policy_hook_chain.py` runs two gates in order. The first gate that says LOCAL wins:

1. **Efficiency**: short and simple questions stay local; long questions, or questions with a
   complexity keyword, can go to SOTA.
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
| `ner.context_entities` | v0.4.0 | `[]` | These entity types (for example `NRP`, `LOCATION`) count only when the text also has an identifier: structured personal data (card, IBAN, email, phone...) or another entity that counts. "European banks in Italy" names nobody |

Ignored entities stay in the log with weight 0, for example `NRP@0.85(no id):0.00` or
`PERSON@0.85(<2 words):0.00`, so the log shows what the engine saw and why it did not count it.

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
| `router_requests_total` | counter | `routed_to`, `decided_by`, `team` | One per decision. `decided_by`: `efficiency`, `privacy`, `tiering`, `all-sota`, `fail-closed`. `team` is a team named in the policies, `none` or `other` |
| `router_privacy_score` | histogram | `team` (threshold key) | Privacy score of the requests that reached the privacy gate |
| `router_sota_budget_used_tokens` | gauge | | SOTA tokens counted by the budget of the efficiency gate, **per pod** |

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
