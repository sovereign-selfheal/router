"""chain routing policy (Phase 4a) — LiteLLM async_pre_call_hook.

Composes the routing gates with UNION-OF-LOCAL semantics: the gates are evaluated
in order and the FIRST gate that returns LOCAL short-circuits the decision to
LOCAL. A request reaches SOTA only if EVERY gate returns SOTA.

    gate_order: [efficiency, privacy]        (see policy/chain.yaml)

  * efficiency = merged cost + complexity gate. LOCAL for short/simple requests,
                 or when the cumulative SOTA token budget is exhausted; SOTA for
                 long/complex ones. Reads the user question (last user turn), since
                 length/complexity is a property of the actual ask. Replaces the two
                 overlapping cost/complexity policies with one gate.
  * privacy    = the hardened privacy-plus engine (privacy_scoring.PrivacyScorer,
                 A+B+C1[+C2]). WHOLE-PAYLOAD scan, fail-closed. Loads its config from
                 policy/privacy-plus.yaml (single source of truth for privacy tuning).
                 Because privacy is the LAST gate before SOTA, NO payload reaches SOTA
                 without clearing the privacy scan — the sovereignty guarantee.

Opt-in and NON-BREAKING: selected only when LiteLLM runs with config-chain.yaml
(callbacks: ["policy_hook_chain.proxy_handler_instance"]). The shipped policy_hook.py,
policy_hook_privacy_plus.py and every shipped config/policy stay byte-identical.

FAIL-CLOSED: any unexpected error routes LOCAL (the safe side for a chain carrying a
privacy control; policy_hook.py fails OPEN — this one must not). Emits the SAME
`[policy-router] {...}` log line and metadata.routing_decision as the other hooks,
plus a per-gate `chain` trace so `oc logs | grep policy-router` and the verify plays
keep working — and the demo can show *why* a request went where it went.

Decision stays BINARY LOCAL/SOTA (no redaction).

OBSERVABILITY (never on the decision path): with LiteLLM's `otel` callback on, the hook
opens a `router.chain` span under the proxy request span, one `gate.<name>` span per gate
and, through privacy_scoring, one `presidio.analyze` span per Presidio call. The router
metrics (prometheus_client) are served on ROUTER_METRICS_PORT (default 9091, 0 = off),
together with any other metric of the process registry (LiteLLM's `prometheus` callback).
Every tracing or metrics error is logged and ignored: the decision and fail-closed are
the same with or without OpenTelemetry and prometheus_client.
"""

import os
import re
import threading

import yaml
from litellm.integrations.custom_logger import CustomLogger

from privacy_scoring import PrivacyScorer, annotate_current_span, safe_span

try:  # prometheus_client is in the LiteLLM image; optional so import never breaks the proxy
    import prometheus_client
except Exception:  # pragma: no cover - defensive
    prometheus_client = None

POLICY_DIR = os.environ.get("POLICY_DIR", "/app/litellm")


def _load_policy(name):
    with open(os.path.join(POLICY_DIR, f"{name}.yaml")) as fh:
        return yaml.safe_load(fh) or {}



def _norm_model_id(value):
    """Reduce a model id to bare alphanumerics for tolerant comparison.

    The id seen at response time is either what the PROVIDER reports
    ("qwen38-27b-fp8") or the CONFIGURED id ("openai/Qwen3.8-27B"). Normalising
    both sides lets a single token ("qwen38-27b") match either form, instead of
    failing on a stray dot or slash.
    """
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


# -- metrics: never on the decision path ----------------------------------------
def _metrics_error(exc):
    print(f"[policy-router] metrics error (ignored): {type(exc).__name__}: {exc}", flush=True)


def _metric(kind, name, documentation, labels=(), **kwargs):
    """Create a metric in the default registry, or reuse it if the module is loaded again."""
    if prometheus_client is None:
        return None
    try:
        return getattr(prometheus_client, kind)(name, documentation, labels, **kwargs)
    except ValueError:  # already registered by an earlier import of this module
        registry = prometheus_client.REGISTRY
        return getattr(registry, "_names_to_collectors", {}).get(name)
    except Exception as exc:
        _metrics_error(exc)
        return None


REQUESTS = _metric("Counter", "router_requests",
                   "Routing decisions of the policy hook.", ("routed_to", "decided_by", "team"))
PRIVACY_SCORE = _metric("Histogram", "router_privacy_score",
                        "Privacy score of the requests that reached the privacy gate.", ("team",),
                        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
SOTA_BUDGET_USED = _metric("Gauge", "router_sota_budget_used_tokens",
                           "SOTA tokens counted by the efficiency gate budget (this pod only).")


def _start_metrics_server():
    """Serve the default registry on ROUTER_METRICS_PORT, once per process.

    Returns the port, or None when the server is off or cannot start. A bind error never
    breaks the proxy: the router works without metrics. The "started" mark is kept on the
    prometheus_client module, so a second import of this file does not bind again.
    """
    if prometheus_client is None:
        return None
    started = getattr(prometheus_client, "_policy_router_metrics_port", None)
    if started:
        return started
    try:
        port = int(os.environ.get("ROUTER_METRICS_PORT", "9091") or 0)
        if port <= 0:
            return None
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            print("[policy-router] metrics server off: PROMETHEUS_MULTIPROC_DIR is set", flush=True)
            return None
        prometheus_client.start_http_server(port)
        prometheus_client._policy_router_metrics_port = port
        print(f"[policy-router] metrics on :{port}/metrics", flush=True)
        return port
    except Exception as exc:
        _metrics_error(exc)
        return None


class ChainRouter(CustomLogger):
    def __init__(self):
        super().__init__()
        self.policy = _load_policy("chain")
        self.local_model = self.policy.get("local_model", "local-fast")
        self.sota_model = self.policy.get("sota_model", "sota-smart")
        self.gate_order = self.policy.get("gate_order", ["efficiency", "privacy"])
        self.tiering = self.policy.get("tiering", {}) or {}
        # Backstop used ONLY when the routing metadata is lost. Matched against the
        # model id seen at response time, which may be the id the PROVIDER reports
        # ("qwen38-27b-fp8") or the CONFIGURED id ("openai/Qwen3.8-27B") — both are
        # normalised to bare alphanumerics, so one token matches either form.
        # Accepts a comma-separated string or a YAML list. Set it to "" (or unset the
        # env var AND clear the policy key) to disable; the primary routed_to check
        # keeps working either way. NOTE: SOTA_SERVED_MATCH is an escape hatch only —
        # no manifest sets it today, the value normally comes from the policy file.
        _ssm = os.environ.get("SOTA_SERVED_MATCH")
        if _ssm is None:
            _ssm = self.policy.get("sota_served_match") or ""
        _parts = list(_ssm) if isinstance(_ssm, (list, tuple)) else str(_ssm).split(",")
        self.sota_served_match = [n for n in (_norm_model_id(t) for t in _parts) if n]

        # -- efficiency gate (merged cost + complexity) -------------------------
        eff = self.policy.get("efficiency", {}) or {}
        self.max_prompt_chars = int(eff.get("max_prompt_chars_for_local", 280))
        self.simple_max_words = int(eff.get("simple_max_words", 40))
        self.complex_keywords = [k.lower() for k in eff.get("complex_keywords", [])]
        self.sota_token_budget = int(eff.get("sota_token_budget", 0))
        self._lock = threading.Lock()
        self._sota_tokens_used = 0

        # -- privacy gate: reuse the privacy-plus engine AND its config ---------
        # Single source of truth: privacy tuning lives in policy/privacy-plus.yaml,
        # not duplicated here. The scorer honours the CLASSIFIER_ENABLED env, so C2
        # is inherited for free (default OFF).
        privacy_cfg = self.policy.get("privacy_policy_file", "privacy-plus")
        self.privacy_policy = _load_policy(privacy_cfg)
        self.threshold_default = float(self.privacy_policy.get("threshold", 0.5))
        self.thresholds = self.privacy_policy.get("thresholds", {}) or {}
        self.scorer = PrivacyScorer(self.privacy_policy)
        # Bounded values of the `team` metric label: the teams named in the policies.
        self._known_teams = {
            t for t in list(self.thresholds) + list(self.tiering) if not str(t).startswith("_")
        }

    # -- text/context helpers --------------------------------------------------
    @staticmethod
    def _last_user_text(data):
        """The actual user question (last user turn) — drives the efficiency gate."""
        for msg in reversed(data.get("messages") or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # multimodal content parts
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
        return ""

    @staticmethod
    def _whole_payload(data):
        """System + every user turn + tool/function arguments — drives the privacy
        gate (a privacy control must never look at the last message only)."""
        parts = []
        for msg in data.get("messages") or []:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
            for call in msg.get("tool_calls") or []:
                args = (call.get("function") or {}).get("arguments")
                if isinstance(args, str):
                    parts.append(args)
            fn_call = msg.get("function_call") or {}
            if isinstance(fn_call.get("arguments"), str):
                parts.append(fn_call["arguments"])
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _team(data):
        for src in ("metadata", "proxy_server_request"):
            headers = (data.get(src) or {}).get("headers") or {}
            team = headers.get("x-team") or headers.get("X-Team")
            if team:
                return team.strip().lower()
        return None

    def _threshold_for(self, team):
        if team and team in self.thresholds:
            return float(self.thresholds[team]), team
        if "_default" in self.thresholds:
            return float(self.thresholds["_default"]), "_default"
        return self.threshold_default, "_default"

    # -- gates: each returns (verdict, reason) with verdict in {"local","sota"} --
    def _gate_efficiency(self, data):
        text = self._last_user_text(data)
        if self.sota_token_budget > 0:
            with self._lock:
                used = self._sota_tokens_used
            if used >= self.sota_token_budget:
                return "local", (
                    f"efficiency: SOTA budget exhausted ({used}/{self.sota_token_budget} tok)"
                )
        lowered = text.lower()
        complex_kw = next((k for k in self.complex_keywords if k in lowered), None)
        words = len(text.split())
        short = len(text) <= self.max_prompt_chars
        simple = words <= self.simple_max_words
        if short and simple and not complex_kw:
            return "local", f"efficiency: short/simple ({len(text)} chars, {words} words)"
        why = []
        if not short:
            why.append(f"{len(text)}>{self.max_prompt_chars} chars")
        if not simple:
            why.append(f"{words}>{self.simple_max_words} words")
        if complex_kw:
            why.append(f"keyword '{complex_kw}'")
        return "sota", "efficiency: long/complex (" + ", ".join(why) + ")"

    async def _gate_privacy(self, data):
        text = self._whole_payload(data)
        team = self._team(data)
        threshold, team_key = self._threshold_for(team)
        score, signals = await self.scorer.score(text, threshold)
        breakdown = self.scorer.format_breakdown(signals)
        annotate_current_span({
            "privacy.score": round(score, 4),
            "privacy.threshold": threshold,
            "privacy.team_key": team_key,
            "privacy.signals": breakdown,
        })
        self._observe_privacy_score(score, team_key)
        if score >= threshold:
            return "local", (
                f"privacy: score {score:.2f} ({breakdown}) >= {threshold:.2f}[team={team_key}]"
            )
        return "sota", f"privacy: score {score:.2f} ({breakdown}) < {threshold:.2f}[team={team_key}]"

    def _apply_tiering(self, target, team):
        """Per-team guardrail: a team capped to LOCAL never reaches SOTA. Can only
        pull the decision toward LOCAL, never toward SOTA."""
        if not team or not self.tiering:
            return target, None
        allowed = self.tiering.get(team, self.tiering.get("_default", []))
        if allowed and target not in allowed:
            return self.local_model, (
                f"tiering: team '{team}' not allowed '{target}' -> LOCAL"
            )
        return target, None

    # -- observability helpers: they log and ignore their own errors --------------
    def _team_label(self, team):
        if not team:
            return "none"
        return team if team in self._known_teams else "other"

    def _observe_privacy_score(self, score, team_key):
        try:
            if PRIVACY_SCORE is not None:
                PRIVACY_SCORE.labels(team=team_key).observe(score)
        except Exception as exc:
            _metrics_error(exc)

    def _count_decision(self, decision, team):
        try:
            if REQUESTS is not None:
                REQUESTS.labels(
                    routed_to=decision.get("routed_to") or "unknown",
                    decided_by=decision.get("decided_by") or "unknown",
                    team=self._team_label(team),
                ).inc()
        except Exception as exc:
            _metrics_error(exc)

    @staticmethod
    def _parent_span(user_api_key_dict, data):
        """The proxy request span that LiteLLM creates when its `otel` callback is on."""
        try:
            md = data.get("metadata") or {}
            return md.get("litellm_parent_otel_span") or getattr(
                user_api_key_dict, "parent_otel_span", None
            )
        except Exception:
            return None

    # -- LiteLLM hooks ---------------------------------------------------------
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        team = None
        try:
            if call_type not in ("completion", "acompletion", "text_completion"):
                return data
            requested = data.get("model")
            team = self._team(data)

            with safe_span("router.chain", parent=self._parent_span(user_api_key_dict, data)) as chain_span:
                # Evaluate gates in order; first LOCAL wins (short-circuit). SOTA only if
                # every gate says SOTA. Privacy, being last, is the egress guard.
                target = self.sota_model
                deciding = "all-sota"
                trace = []
                for gate in self.gate_order:
                    if gate not in ("efficiency", "privacy"):
                        continue  # unknown gate name in config -> skip, don't crash
                    with safe_span(f"gate.{gate}") as gate_span:
                        if gate == "efficiency":
                            verdict, reason = self._gate_efficiency(data)
                        else:
                            verdict, reason = await self._gate_privacy(data)
                        gate_span.set("gate.verdict", verdict)
                        gate_span.set("gate.reason", reason)
                    trace.append(f"{reason} -> {verdict.upper()}")
                    if verdict == "local":
                        target, deciding = self.local_model, gate
                        break

                target, tier_reason = self._apply_tiering(target, team)
                data["model"] = target

                decision = {
                    "policy": "chain",
                    "requested": requested,
                    "routed_to": target,
                    "decided_by": "tiering" if tier_reason else deciding,
                    "chain": trace,
                    "reason": tier_reason or (trace[-1] if trace else "no-gate"),
                    "team": team,
                }
                trace_id = chain_span.trace_id()
                if trace_id:
                    decision["trace_id"] = trace_id  # additive: find the trace from the log line
                chain_span.set("route.requested", requested)
                chain_span.set("route.target", target)
                chain_span.set("route.decided_by", decision["decided_by"])
                chain_span.set("route.team", team)
                chain_span.set("route.reason", decision["reason"])
            data.setdefault("metadata", {})["routing_decision"] = decision
            print(f"[policy-router] {decision}", flush=True)
        except Exception as exc:
            # FAIL-CLOSED: on any unexpected error, keep the payload LOCAL.
            data["model"] = self.local_model
            decision = {
                "policy": "chain",
                "routed_to": self.local_model,
                "reason": f"chain: fail-closed on error ({exc}) -> LOCAL",
            }
            data.setdefault("metadata", {})["routing_decision"] = decision
            print(f"[policy-router] {decision}", flush=True)
            decision = dict(decision, decided_by="fail-closed")
        self._count_decision(decision, team)
        return data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Accumulate SOTA token usage to drive the efficiency gate's budget guard.
        try:
            if self.sota_token_budget <= 0:
                return
            md = (kwargs.get("litellm_params") or {}).get("metadata") or {}
            decision = (
                md.get("routing_decision")
                or (kwargs.get("metadata") or {}).get("routing_decision")
                or {}
            )
            served = str(getattr(response_obj, "model", "") or kwargs.get("model") or "")
            _served_n = _norm_model_id(served)
            is_sota = decision.get("routed_to") == self.sota_model or any(
                tok in _served_n for tok in self.sota_served_match
            )
            if not is_sota:
                return
            usage = getattr(response_obj, "usage", None)
            total = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
            if total:
                with self._lock:
                    self._sota_tokens_used += total
                    used = self._sota_tokens_used
                try:
                    if SOTA_BUDGET_USED is not None:
                        SOTA_BUDGET_USED.set(used)
                except Exception as exc:
                    _metrics_error(exc)
                print(f"[policy-router] SOTA budget: {used}/{self.sota_token_budget} tokens used", flush=True)
        except Exception as exc:
            print(f"[policy-router] success-event error: {exc}", flush=True)


_start_metrics_server()
proxy_handler_instance = ChainRouter()
