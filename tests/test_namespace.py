"""Namespace policy (v0.11.0): a restricted namespace keeps the request LOCAL."""

import ast
import asyncio
import time

import namespace_policy
import policy_hook_chain
import pytest
from conftest import FakePresidio
from namespace_policy import NamespaceLabels, NamespacePolicy, hint_names, pop_hint, scan_names
from prometheus_client import REGISTRY

LONG_BENIGN = ("Analyze the trade-offs between event sourcing and a classic CRUD model for an "
               "order management system, with a short example.")
LABELS = {"payments": "restricted", "agentic-triage": "public", "typo-ns": "restriced"}
CFG = {"scan": True, "hint": True, "label": "sovereign-selfheal.io/data-class",
       "hint_field": "selfheal_namespaces", "refresh_s": 5}


def make_router(cfg=CFG, labels=LABELS, fail=False):
    """A ChainRouter with the namespace policy on and a fake Kubernetes API (no thread)."""
    r = policy_hook_chain.ChainRouter()
    r.scorer._lang_detector = lambda _text: ("en", 0.99)
    r.scorer._query_presidio = FakePresidio()

    def fetch():
        if fail:
            raise ConnectionError("API down")
        return dict(labels)

    r.namespaces = NamespacePolicy(cfg, fetch=fetch, on_change=r._observe_namespace_labels,
                                   start=False)
    if r.namespaces.enabled:
        r.namespaces.labels.refresh()
        r._observe_namespace_labels(r.namespaces.labels)
    return r


def route(router, text, hint=None, team="research", call_type="acompletion", extra=None):
    data = {"model": "auto", "messages": [{"role": "user", "content": text}],
            "metadata": {"headers": {"x-team": team}}}
    if hint is not None:
        data["selfheal_namespaces"] = hint
    data.update(extra or {})
    out = asyncio.run(router.async_pre_call_hook(None, None, data, call_type))
    return out, (out.get("metadata") or {}).get("routing_decision")


# ------------------------------------------------------------------ scan and hint parsing
def test_scan_finds_structured_mentions():
    text = (
        'rate(http_server_requests_seconds_count{namespace="payments",status=~"5.."}[5m])\n'
        '{"namespace":"Agentic-Triage","pod":"x"} exported_namespace="local-models"\n'
        "curl http://prometheus-mcp-server.obs-a.svc:8080/mcp\n"
        "litellm.maas-routing.svc.cluster.local\n"
        "oc get pods -n ns-b; oc logs --namespace=ns-c x; GET /api/v1/namespaces/kube-system/pods\n"
        'namespace: yaml-ns  namespace!="excluded" namespace=~"regex-a|b"'
    )
    assert scan_names(text) >= {
        "payments", "agentic-triage", "local-models", "obs-a", "maas-routing", "ns-b", "ns-c",
        "kube-system", "yaml-ns", "excluded", "regex-a"}


def test_scan_reads_the_alert_labels_of_the_translator_prompt():
    # Format of ogx-alert-translator (app/prompt_builder.py): one bullet per label.
    text = "**Labels:**\n- `alertname`: QuarkusBuggyAppHighErrorRate\n- `namespace`: payments\n"
    assert scan_names(text) == {"payments"}
    assert scan_names("- `namespace`: `payments`") == {"payments"}


def test_scan_ignores_prose_and_other_keys():
    text = 'the payments team; subnamespace="nope"; mynamespace: nope2; payments-svc is up'
    assert scan_names(text) == set()


def test_scan_is_fast_on_large_contexts():
    text = ('level=info msg="GET /orders 200" pod=api-7 ' * 9500
            + 'rate(x{namespace="payments"}[5m])')
    assert len(text) > 400000
    start = time.perf_counter()
    assert "payments" in scan_names(text)
    assert time.perf_counter() - start < 0.5


def test_hint_names_are_validated():
    assert hint_names("Payments") == ["payments"]
    assert hint_names(["payments", "payments", " a-b ", 3, "", "Not_Valid", "x" * 64]) == [
        "payments", "a-b"]
    assert hint_names({"payments": 1}) == []
    assert len(hint_names([f"ns-{i}" for i in range(50)])) == 20


def test_pop_hint_removes_the_field_everywhere():
    data = {"selfheal_namespaces": ["a"], "extra_body": {"selfheal_namespaces": ["b"], "k": 1}}
    assert pop_hint(data, "selfheal_namespaces") == ["a"]
    assert "selfheal_namespaces" not in data
    assert data["extra_body"] == {"k": 1}
    nested = {"extra_body": {"selfheal_namespaces": ["b"]}}
    assert pop_hint(nested, "selfheal_namespaces") == ["b"]
    assert pop_hint({"model": "auto"}, "selfheal_namespaces") is None


# ------------------------------------------------------------------ off by default
def test_policy_is_off_by_default_and_the_hint_is_still_removed():
    r = policy_hook_chain.ChainRouter()
    r.scorer._lang_detector = lambda _text: ("en", 0.99)
    r.scorer._query_presidio = FakePresidio()
    assert r.namespaces.enabled is False
    assert r.namespaces.labels is None  # no API call, no thread
    out, decision = route(r, LONG_BENIGN + ' namespace="payments"', hint=["payments"])
    assert "selfheal_namespaces" not in out
    assert out["model"] == "sota-smart"
    assert "ns_restricted" not in decision
    assert decision["decided_by"] == "all-sota"


def test_hint_is_removed_for_other_call_types():
    r = make_router()
    out, decision = route(r, LONG_BENIGN, hint=["payments"], call_type="embeddings")
    assert "selfheal_namespaces" not in out
    assert out["model"] == "auto"
    assert decision is None


def test_env_overrides_the_policy(monkeypatch):
    monkeypatch.setenv("NAMESPACE_SCAN_ENABLED", "1")
    monkeypatch.setenv("NAMESPACE_HINT_ENABLED", "")
    p = NamespacePolicy({"scan": False, "hint": True}, fetch=dict, start=False)
    assert (p.scan_enabled, p.hint_enabled, p.enabled) == (True, True, True)
    monkeypatch.setenv("NAMESPACE_HINT_ENABLED", "off")
    monkeypatch.setenv("NAMESPACE_SCAN_ENABLED", "no")
    p = NamespacePolicy({"scan": True, "hint": True}, fetch=dict, start=False)
    assert p.enabled is False


# ------------------------------------------------------------------ decisions
def test_restricted_hint_routes_local_without_gates():
    r = make_router()
    out, decision = route(r, LONG_BENIGN, hint=["payments"])
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "namespace"
    assert decision["ns_restricted"] == ["payments"]
    assert decision["ns_source"] == "hint"
    assert decision["chain"] == ["namespace: restricted payments (found by hint) -> LOCAL"]
    assert r.scorer._query_presidio.calls == []  # no detector ran
    assert "ner_timeout_s" not in decision


def test_scan_restricts_a_public_hint():
    r = make_router()
    text = LONG_BENIGN + '\nTool output: up{namespace="payments", pod="api-1"} 1'
    out, decision = route(r, text, hint=["agentic-triage"])
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "namespace"
    assert decision["ns_source"] == "scan"
    assert decision["namespaces"] == ["agentic-triage", "payments"]


def test_scan_alone_finds_a_restricted_namespace():
    r = make_router()
    _out, decision = route(r, LONG_BENIGN + " curl http://api.payments.svc:8080/health")
    assert decision["routed_to"] == "local-fast"
    assert decision["ns_source"] == "scan"


def test_public_and_unlabelled_namespaces_keep_the_normal_routing():
    r = make_router()
    out, decision = route(r, LONG_BENIGN + ' namespace="agentic-triage" namespace="other"',
                          hint=["agentic-triage", "not-labelled"])
    assert out["model"] == "sota-smart"
    assert decision["decided_by"] == "all-sota"
    assert decision["ns_restricted"] == []
    assert decision["ns_source"] == "hint+scan"
    assert decision["namespaces"] == ["agentic-triage", "not-labelled"]
    assert decision["chain"][0].startswith("efficiency:")
    assert r.scorer._query_presidio.calls  # the privacy gate ran


def test_short_public_question_still_follows_the_efficiency_gate():
    r = make_router()
    _out, decision = route(r, "What is a pod?", hint=["agentic-triage"])
    assert decision["routed_to"] == "local-fast"
    assert decision["decided_by"] == "efficiency"


def test_unknown_label_value_keeps_the_normal_routing():
    r = make_router()
    out, decision = route(r, LONG_BENIGN, hint=["typo-ns"])
    assert out["model"] == "sota-smart"
    assert decision["ns_restricted"] == []
    assert REGISTRY.get_sample_value("router_namespace_labels", {"state": "unknown"}) == 1
    assert REGISTRY.get_sample_value("router_namespace_labels", {"state": "restricted"}) == 1
    assert REGISTRY.get_sample_value("router_namespace_labels", {"state": "public"}) == 1


def test_hint_switch_off_ignores_the_hint_but_the_scan_works():
    r = make_router(cfg=dict(CFG, hint=False))
    out, decision = route(r, LONG_BENIGN, hint=["payments"])
    assert "selfheal_namespaces" not in out
    assert out["model"] == "sota-smart"
    _out, decision = route(r, LONG_BENIGN + ' namespace="payments"')
    assert decision["decided_by"] == "namespace"


def test_scan_switch_off_ignores_the_text():
    r = make_router(cfg=dict(CFG, scan=False))
    _out, decision = route(r, LONG_BENIGN + ' namespace="payments"')
    assert decision["routed_to"] == "sota-smart"
    assert decision["ns_source"] == "none"


def test_labels_never_read_keep_the_normal_routing():
    r = make_router(fail=True)
    out, decision = route(r, LONG_BENIGN, hint=["payments"])
    assert out["model"] == "sota-smart"
    assert decision["ns_labels_loaded"] is False
    assert decision["chain"][0].startswith("efficiency:")
    assert "ConnectionError" in r.namespaces.labels.last_error
    assert REGISTRY.get_sample_value("router_namespace_labels_loaded") == 0


def test_failed_refresh_keeps_the_last_list():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        if calls["n"] > 1:
            raise TimeoutError("API slow")
        return {"payments": "restricted"}

    labels = NamespaceLabels("k", fetch=fetch)
    assert labels.refresh() is True
    assert labels.refresh() is False
    assert labels.restricted() == {"payments"}
    assert labels.loaded is True
    assert "TimeoutError" in labels.last_error


def test_label_change_is_seen_at_the_next_refresh():
    current = {"payments": "restricted"}
    r = make_router()
    r.namespaces.labels._fetch = lambda: dict(current)
    r.namespaces.labels.refresh()
    assert route(r, LONG_BENIGN, hint=["payments"])[1]["routed_to"] == "local-fast"
    current["payments"] = "public"
    r.namespaces.labels.refresh()
    assert route(r, LONG_BENIGN, hint=["payments"])[1]["routed_to"] == "sota-smart"


def test_legal_tiering_still_applies():
    r = make_router()
    _out, decision = route(r, LONG_BENIGN, hint=["agentic-triage"], team="legal")
    assert decision["routed_to"] == "local-fast"
    assert decision["decided_by"] == "tiering"


def test_namespace_error_fails_closed(monkeypatch):
    r = make_router()

    def boom(*_args):
        raise RuntimeError("bug")

    monkeypatch.setattr(r.namespaces, "evaluate", boom)
    out, decision = route(r, LONG_BENIGN, hint=["agentic-triage"])
    assert out["model"] == "local-fast"
    assert "fail-closed" in decision["reason"]


# ------------------------------------------------------------------ log line and metrics
def test_log_line_has_the_namespace_fields(capsys):
    r = make_router()
    route(r, LONG_BENIGN, hint=["payments"])
    lines = capsys.readouterr().out.splitlines()
    line = [x for x in lines if x.startswith("[policy-router] {")][-1]
    decision = ast.literal_eval(line.split("[policy-router] ", 1)[1])
    keys = {"policy", "requested", "routed_to", "decided_by", "chain", "reason", "team"}
    assert keys <= set(decision)
    assert decision["decided_by"] == "namespace"
    assert decision["namespaces"] == ["payments"]
    assert decision["ns_labels_loaded"] is True


def test_namespace_decisions_metric():
    def count(ns, routed_to, source):
        return REGISTRY.get_sample_value(
            "router_namespace_decisions_total",
            {"namespace": ns, "routed_to": routed_to, "source": source}) or 0.0

    r = make_router()
    before_r = count("payments", "local-fast", "hint")
    before_n = count("none", "sota-smart", "hint")
    route(r, LONG_BENIGN, hint=["payments"])
    route(r, LONG_BENIGN, hint=["agentic-triage"])
    assert count("payments", "local-fast", "hint") == before_r + 1
    assert count("none", "sota-smart", "hint") == before_n + 1
    assert REGISTRY.get_sample_value(
        "router_requests_total",
        {"routed_to": "local-fast", "decided_by": "namespace", "team": "research"}) >= 1


# ------------------------------------------------------------------ Kubernetes API read
def test_fetch_from_api_reads_the_labels(monkeypatch, tmp_path):
    (tmp_path / "token").write_text("tok\n")
    (tmp_path / "ca.crt").write_text("unused")
    monkeypatch.setattr(namespace_policy, "SA_DIR", str(tmp_path))
    monkeypatch.setattr(namespace_policy.ssl, "create_default_context", lambda cafile: cafile)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "fd00::1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"items": [
                {"metadata": {"name": "payments", "labels": {"k": "Restricted"}}},
                {"metadata": {"name": "agentic-triage", "labels": {"k": "public"}}},
            ]}

    def fake_get(url, params, headers, verify, timeout):
        seen.update(url=url, params=params, auth=headers["Authorization"], verify=verify)
        return Resp()

    monkeypatch.setattr(namespace_policy.httpx, "get", fake_get)
    labels = NamespaceLabels("k")
    assert labels.refresh() is True
    assert labels.labels() == {"payments": "restricted", "agentic-triage": "public"}
    assert seen["url"] == "https://[fd00::1]:443/api/v1/namespaces"
    assert seen["params"] == {"labelSelector": "k"}
    assert seen["auth"] == "Bearer tok"
    assert seen["verify"].endswith("ca.crt")


@pytest.mark.parametrize("value", [None, ""])
def test_env_flag_unset(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("NAMESPACE_SCAN_ENABLED", raising=False)
    else:
        monkeypatch.setenv("NAMESPACE_SCAN_ENABLED", value)
    assert namespace_policy.env_flag("NAMESPACE_SCAN_ENABLED") is None
