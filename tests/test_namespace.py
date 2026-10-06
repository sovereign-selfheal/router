"""Namespace policy (v0.11.0; scan fixes v0.11.1): a restricted namespace stays LOCAL."""

import ast
import asyncio
import json
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


def test_scan_reads_json_encoded_tool_call_arguments():
    # Tool call arguments are JSON strings: the quotes of PromQL are escaped (v0.11.1).
    query = 'sum(rate(http_server_requests_seconds_count{namespace="payments",status=~"5.."}[5m]))'
    args = json.dumps({"query": query})
    assert '\\"payments\\"' in args
    assert scan_names(args) == {"payments"}
    assert scan_names(json.dumps({"payload": args})) == {"payments"}  # encoded twice


def test_scan_reads_json_argv_arrays():
    argv = json.dumps({"command": ["oc", "get", "pods", "-n", "payments"]})
    assert scan_names(argv) == {"payments"}
    assert scan_names(json.dumps(["kubectl", "logs", "--namespace", "ns-b", "x"])) == {"ns-b"}
    assert scan_names("['oc','get','pods','-n','ns-c']") == {"ns-c"}


def test_scan_reads_every_alternative_of_a_regex_matcher():
    assert scan_names('rate(m{namespace=~"agentic-triage|payments"}[5m])') == {
        "agentic-triage", "payments"}
    assert scan_names('m{namespace=~"(payments)"}') == {"payments"}
    assert scan_names('m{namespace=~"^(?:payments|ns-b)$"}') == {"payments", "ns-b"}
    assert scan_names('m{namespace!~"ns-a|ns-b"}') == {"ns-a", "ns-b"}  # stricter, as for !=


def test_scan_expands_wildcard_prefixes_against_known_names():
    known = {"payments", "payroll", "agentic-triage"}
    assert scan_names('m{namespace=~"pay.*"}', known=known) >= {"payments", "payroll"}
    assert "agentic-triage" not in scan_names('m{namespace=~"pay.*"}', known=known)
    assert scan_names('m{namespace=~"agentic-triage|pay.+"}', known=known) >= {
        "agentic-triage", "payments", "payroll"}
    # A match-all value names no namespace (like a query without namespace).
    assert scan_names('m{namespace=~".*"}', known=known) == set()
    assert scan_names('m{namespace=~".+"}', known=known) == set()


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
    # v0.11.1: every valid name is checked (no cap at 20); only the input is bounded.
    assert len(hint_names([f"ns-{i}" for i in range(50)])) == 50
    assert len(hint_names([f"ns-{i}" for i in range(900)])) == namespace_policy.MAX_HINT_ITEMS


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


def test_tool_call_arguments_reach_the_scan():
    # An investigation started by a human ("why is payments failing?"), hint off: the only
    # structured mention is the PromQL in the JSON arguments of a tool call (v0.11.1).
    cfg = dict(CFG, hint=False)
    r = make_router(cfg=cfg)
    query = 'sum(rate(http_server_requests_seconds_count{namespace="payments",status=~"5.."}[5m]))'
    extra = {"messages": [
        {"role": "user", "content": LONG_BENIGN},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "query_prometheus", "arguments": json.dumps({"query": query})}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": '{"status":"success","data":{"result":[]}}'},
    ]}
    out, decision = route(r, "unused", extra=extra)
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "namespace"
    assert decision["ns_restricted"] == ["payments"]
    assert decision["ns_source"] == "scan"


def test_regex_matcher_with_two_namespaces_routes_local():
    r = make_router()
    _out, decision = route(r, LONG_BENIGN + ' rate(m{namespace=~"agentic-triage|payments"}[5m])')
    assert decision["routed_to"] == "local-fast"
    assert decision["ns_restricted"] == ["payments"]
    _out, decision = route(r, LONG_BENIGN + ' rate(m{namespace=~"pay.*"}[5m])')
    assert decision["routed_to"] == "local-fast"
    assert decision["ns_restricted"] == ["payments"]


def test_restricted_hint_after_many_names_is_seen():
    r = make_router()
    hint = [f"ns-{i}" for i in range(30)] + ["payments"]
    _out, decision = route(r, LONG_BENIGN, hint=hint)
    assert decision["routed_to"] == "local-fast"
    assert decision["ns_restricted"] == ["payments"]
    assert len(decision["namespaces"]) == namespace_policy.MAX_LOG_NAMES


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


def test_restricted_series_start_at_zero():
    labels = {"zero-ns": "restricted"}
    make_router(labels=labels)
    value = REGISTRY.get_sample_value(
        "router_namespace_decisions_total",
        {"target_namespace": "zero-ns", "routed_to": "local-fast", "source": "hint"})
    assert value == 0.0
    # The most common series: no restricted namespace, nothing found (v0.11.1).
    for routed_to in ("local-fast", "sota-smart"):
        assert REGISTRY.get_sample_value(
            "router_namespace_decisions_total",
            {"target_namespace": "none", "routed_to": routed_to, "source": "none"}) is not None


def test_namespace_decisions_metric():
    def count(ns, routed_to, source):
        return REGISTRY.get_sample_value(
            "router_namespace_decisions_total",
            {"target_namespace": ns, "routed_to": routed_to, "source": source}) or 0.0

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
    clients, calls = [], []

    class Resp:
        def raise_for_status(self):
            if calls[-1].get("fail"):
                raise ConnectionError("API down")

        def json(self):
            return {"items": [
                {"metadata": {"name": "payments", "labels": {"k": "Restricted"}}},
                {"metadata": {"name": "agentic-triage", "labels": {"k": "public"}}},
            ]}

    class FakeClient:
        fail_next = False

        def __init__(self, base_url, verify, timeout):
            self.base_url, self.verify, self.closed = base_url, verify, False
            clients.append(self)

        def get(self, path, params, headers):
            calls.append({"path": path, "params": params, "auth": headers["Authorization"],
                          "fail": FakeClient.fail_next})
            return Resp()

        def close(self):
            self.closed = True

    monkeypatch.setattr(namespace_policy.httpx, "Client", FakeClient)
    labels = NamespaceLabels("k")
    assert labels.refresh() is True
    assert labels.labels() == {"payments": "restricted", "agentic-triage": "public"}
    assert clients[0].base_url == "https://[fd00::1]:443"
    assert clients[0].verify.endswith("ca.crt")
    assert calls[0] == {"path": "/api/v1/namespaces", "params": {"labelSelector": "k"},
                        "auth": "Bearer tok", "fail": False}
    # The TLS client is reused: no new handshake setup at every refresh (v0.11.1).
    (tmp_path / "token").write_text("tok2\n")
    assert labels.refresh() is True
    assert len(clients) == 1 and calls[1]["auth"] == "Bearer tok2"
    # After an error the client is closed and built again at the next refresh.
    FakeClient.fail_next = True
    assert labels.refresh() is False
    assert clients[0].closed is True
    FakeClient.fail_next = False
    assert labels.refresh() is True
    assert len(clients) == 2


@pytest.mark.parametrize("value", [None, ""])
def test_env_flag_unset(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("NAMESPACE_SCAN_ENABLED", raising=False)
    else:
        monkeypatch.setenv("NAMESPACE_SCAN_ENABLED", value)
    assert namespace_policy.env_flag("NAMESPACE_SCAN_ENABLED") is None
