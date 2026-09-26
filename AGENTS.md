# AGENTS.md: `router` repository

Guidance for AI coding agents and humans working in this repo. Read it fully before you change anything.

## 1. Purpose

This repository is the **source of truth of the router code**: the LiteLLM hook that sends each request
to the local model or to the external SOTA model (`litellm/policy_hook_chain.py`), and the privacy engine
it uses (`litellm/privacy_scoring.py`). It also builds the LiteLLM image with fastText,
`quay.io/sovereign-selfheal/router`. The `gitops` repo deploys both. This repo contains no Kubernetes
manifests and no policies.

## 2. Contract with the `gitops` and `presidio` repos

**Ownership. Each item has one owner.**

| Owner | Items |
|---|---|
| `router` (this repo) | Hook code (`litellm/*.py`), unit tests, evaluation sets and baseline, `Containerfile` of the LiteLLM image, the image tags |
| `gitops` | The policies `chain.yaml` and `privacy-plus.yaml` (source of truth), the LiteLLM config `config-chain.yaml`, the ConfigMap with a **copy** of the hook code, the Deployment, the image digest in use |
| `presidio` | The Presidio analyzer image (NER models, API) |

Rules. The same rules are in `gitops/AGENTS.md`: keep both in sync.

1. **One version.** A release is a git tag `vX.Y.Z`. The image `router:vX.Y.Z` and the hook code at
   `vX.Y.Z` go together. In `gitops`, `routerVersion` in `components/litellm-router/values.yaml` names
   the tag, and the image digest must come from the same tag.
2. **Checked copy.** `gitops/components/litellm-router/files/*.py` is byte-identical to `litellm/*.py`
   at `routerVersion`. `gitops/scripts/sync-router-code.sh sync vX.Y.Z` copies the files, and the gitops CI
   checks them (`sync-router-code.sh check`). Never edit the copy in `gitops`: change the code here.
3. **Backward-compatible policy keys.** The code reads its settings from the gitops policies. Every new
   key must have a default that keeps the previous behaviour. `gitops` turns a new key on only after it
   uses the release that reads it. Document every new key in the policy comments of `gitops` and in the
   release PR.
4. **Stable interface.** Do not change these without a new **minor** version, a note in the README and a
   matching change in `gitops`:
   - the log line `[policy-router] {...}`: a Python dict literal with at least `policy`, `requested`,
     `routed_to`, `decided_by`, `chain`, `reason`, `team` (the e2e tests and the demo video read it;
     `tests/test_chain.py::test_log_line_contract` checks it);
   - `metadata.routing_decision` on the request, with the same content;
   - the entry point `policy_hook_chain.proxy_handler_instance` (named in `config-chain.yaml`);
   - the model aliases `local-fast` and `sota-smart` (defaults, overridable in `chain.yaml`);
   - the env vars `POLICY_DIR`, `SOTA_SERVED_MATCH`, `CLASSIFIER_ENABLED`, `CLASSIFIER_GRAY_LOW`,
     `CLASSIFIER_BASE_URL`, `CLASSIFIER_MODEL`, `CLASSIFIER_API_KEY`, `ROUTER_METRICS_PORT` (v0.5.0);
   - the span names `router.chain`, `gate.<name>`, `presidio.analyze` and their attributes, and the
     metric names `router_requests_total`, `router_privacy_score`, `router_sota_budget_used_tokens`
     (v0.5.0; the demo video and `gitops/docs/observability.md` use them);
   - in the image: the model at `/opt/models/lid.176.ftz`, the `litellm` command, a non-root user.
5. **Presidio interface.** The code calls `POST /analyze` with `text`, `language` (`en` or `it`) and the
   `entities` it scores. The entity types come from `ner.entity_weights` in the policy. A change of the
   Presidio model changes the scores: run the evaluation against the new image (see `presidio/AGENTS.md`).

## 3. Layout

```
litellm/                 # hook code, copied by gitops into the LiteLLM ConfigMap
  policy_hook_chain.py   # LiteLLM async_pre_call_hook: efficiency gate, privacy gate, tiering
  privacy_scoring.py     # privacy engine; no LiteLLM import, so it is testable alone
tests/                   # unit tests (fake Presidio, fake LiteLLM); tests/policy/ = copy of the gitops policies
eval/                    # labelled prompt sets, run_eval.py, check_baseline.py, baseline.json
scripts/                 # fetch-lid-model.sh
Containerfile            # LiteLLM + fastText + lid.176.ftz
```

## 4. Conventions

- **Fail-closed.** On any error or doubt the request stays LOCAL. Never add a path that sends a request
  to SOTA when a detector fails.
- **No LiteLLM import in `privacy_scoring.py`**: it must stay testable without LiteLLM.
- **Instrumentation is never on the decision path.** Traces and metrics only observe: every tracing or
  metrics call is wrapped so that its error is logged and ignored, the decision is computed the same
  way with or without them, and fail-closed is unchanged. OpenTelemetry and `prometheus_client` stay
  optional imports (like `httpx` and `fasttext`). Tests prove it (`tests/test_observability.py`).
- **Tests for every change.** A code change comes with unit tests. A change of the routing behaviour also
  shows the evaluation results in the PR. `eval/baseline.json` changes only with `--update`, and the PR
  says why (new leaks are never accepted without a reason).
- **Test policies.** `tests/policy/` is a copy of the gitops policies. Refresh it when the gitops policies
  change; the tests must pass with both the old keys and the new ones.
- **Pins.** The base image is pinned by digest; packages and model files by version and sha256. A
  comment says where and when it was resolved (`# ..., resolved on ghcr.io on 2026-09-25`).
- **Build-time downloads only.** The pod has no egress: everything the image needs is added in the
  `Containerfile`, with a checked hash.
- **Support status.** LiteLLM, Presidio and spaCy are community software, not supported by Red Hat. Say
  so in every document that describes them.
- **Python tools with uv** (`uv sync`, `uv run`), never pip on the host. Change a dependency in
  `pyproject.toml`, run `uv lock`, commit both files. Python 3.13, like the image.
- Comments, docs and commit messages in **English**, level B2/C1: short, clear sentences, no idioms.
  The hook code keeps the style it came with; new code follows `ruff`.

## 5. Release

Quay builds and publishes the image. This repo pushes nothing.

1. Merge on `main` with a green CI.
2. Tag `vX.Y.Z` and push the tag. Quay's build trigger builds it and publishes
   `quay.io/sovereign-selfheal/router:vX.Y.Z`.
3. PR on `gitops`: `scripts/sync-router-code.sh sync vX.Y.Z`, then pin the image digest of the same tag
   (`# tag vX.Y.Z, resolved on quay.io on <date>`). If the release adds policy keys, turn them on in the
   same PR or a later one.

- Semantic versions: **patch** for fixes with the same behaviour and interface; **minor** for a new
  policy key, a change of routing behaviour or of the interface in §2; **major** for a breaking change.
- Every tag builds a new image, even when only the code changed: the image and the code share the tag.
- Never move, delete or reuse a tag. `gitops` never follows a tag automatically.

## 6. Before you open a PR

```bash
uv sync --locked
uv run ruff check .
uv run yamllint .
shellcheck scripts/*.sh
hadolint Containerfile
uv run pytest
uv run eval/check_baseline.py     # with Presidio on localhost:3000 (see README "Evaluation")
```

## 7. Out of scope

- Policies, LiteLLM config, Kubernetes manifests, the ConfigMap → `gitops` repo.
- The Presidio image and its NER models → `presidio` repo.
- Cluster preparation, operators, the Gateway and the Route → `ansible` repo.

## 8. When in doubt

- Prefer the smallest change that keeps the interface in §2 valid.
- Ask before changing the gate order, a default value, the fail-closed behaviour, or the interface in §2.
