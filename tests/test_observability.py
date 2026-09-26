"""Traces and metrics of the policy hook: they show the decision and never change it."""

import ast
import asyncio
import os
import pathlib
import socket
import subprocess
import sys
import types

import policy_hook_chain
import privacy_scoring
import pytest
from conftest import FakePresidio
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import REGISTRY

LONG_BENIGN = ("Analyze the trade-offs between event sourcing and a classic CRUD model for an "
               "order management system, with a short example.")
WITH_CARD = LONG_BENIGN + " Customer card 4111 1111 1111 1111."
ROOT = pathlib.Path(__file__).resolve().parent.parent

# One SDK provider for the whole session (OpenTelemetry sets the global provider only once),
# like LiteLLM does when its `otel` callback is on.
EXPORTER = InMemorySpanExporter()
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    _provider = TracerProvider()
    _provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(_provider)
else:  # pragma: no cover - another module set it first
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(EXPORTER))


@pytest.fixture
def router():
    r = policy_hook_chain.ChainRouter()
    r.scorer._lang_detector = lambda _text: ("en", 0.99)
    r.scorer._query_presidio = FakePresidio()
    EXPORTER.clear()
    return r


def route(router, text, team="research", parent=None):
    metadata = {"headers": {"x-team": team}} if team else {}
    if parent is not None:
        metadata["litellm_parent_otel_span"] = parent
    data = {"model": "auto", "messages": [{"role": "user", "content": text}], "metadata": metadata}
    out = asyncio.run(router.async_pre_call_hook(None, None, data, "acompletion"))
    return out, out["metadata"]["routing_decision"]


def route_under_proxy_span(router, text, team="research"):
    """Like LiteLLM: the hook runs under the proxy request span, which it gets in metadata."""
    proxy = trace.get_tracer("test").start_span("proxy-request")
    out, decision = route(router, text, team, parent=proxy)
    proxy.end()
    spans = {s.name: s for s in EXPORTER.get_finished_spans()}
    return out, decision, spans, proxy


def requests_count(routed_to, decided_by, team):
    return REGISTRY.get_sample_value(
        "router_requests_total",
        {"routed_to": routed_to, "decided_by": decided_by, "team": team}) or 0.0


def test_local_by_privacy_span_tree(router):
    out, decision, spans, proxy = route_under_proxy_span(router, WITH_CARD)
    assert out["model"] == "local-fast" and decision["decided_by"] == "privacy"

    chain = spans["router.chain"]
    assert chain.parent.span_id == proxy.get_span_context().span_id
    assert chain.attributes["route.target"] == "local-fast"
    assert chain.attributes["route.decided_by"] == "privacy"
    assert chain.attributes["route.requested"] == "auto"
    assert chain.attributes["route.team"] == "research"

    eff, priv = spans["gate.efficiency"], spans["gate.privacy"]
    assert eff.parent.span_id == priv.parent.span_id == chain.context.span_id
    assert eff.attributes["gate.verdict"] == "sota"
    assert priv.attributes["gate.verdict"] == "local"
    assert priv.attributes["gate.reason"] == decision["reason"].rsplit(" -> ", 1)[0]
    assert priv.attributes["privacy.score"] >= priv.attributes["privacy.threshold"]
    assert priv.attributes["privacy.team_key"]
    assert "credit_card" in priv.attributes["privacy.signals"].lower()

    presidio = spans["presidio.analyze"]
    assert presidio.parent.span_id == priv.context.span_id
    assert presidio.attributes["presidio.language"] == "en"
    assert presidio.attributes["presidio.entities_found"] == 0

    # The trace id in the decision (and in the log line) finds this trace in Tempo.
    assert decision["trace_id"] == format(chain.context.trace_id, "032x")
    assert chain.context.trace_id == proxy.get_span_context().trace_id


def test_sota_span_tree(router):
    out, decision, spans, _proxy = route_under_proxy_span(router, LONG_BENIGN)
    assert out["model"] == "sota-smart" and decision["decided_by"] == "all-sota"
    assert spans["router.chain"].attributes["route.target"] == "sota-smart"
    assert spans["gate.efficiency"].attributes["gate.verdict"] == "sota"
    assert spans["gate.privacy"].attributes["gate.verdict"] == "sota"


def test_short_circuit_opens_no_privacy_span(router):
    _out, decision, spans, _proxy = route_under_proxy_span(router, "What is 2+2?")
    assert decision["decided_by"] == "efficiency"
    assert "gate.efficiency" in spans and "gate.privacy" not in spans


def test_without_parent_span_the_chain_is_a_root_span(router):
    _out, decision = route(router, LONG_BENIGN)
    chain = {s.name: s for s in EXPORTER.get_finished_spans()}["router.chain"]
    assert chain.parent is None
    assert decision["trace_id"] == format(chain.context.trace_id, "032x")


def test_presidio_error_is_recorded_and_still_fails_closed(router):
    router.scorer._query_presidio = FakePresidio(error=RuntimeError("presidio down"))
    out, decision, spans, _proxy = route_under_proxy_span(router, LONG_BENIGN)
    assert out["model"] == "local-fast" and decision["decided_by"] == "privacy"
    assert spans["presidio.analyze"].status.status_code == trace.StatusCode.ERROR
    assert "error:RuntimeError" in spans["gate.privacy"].attributes["privacy.signals"]


def test_unexpected_error_fails_closed_and_is_counted(router, monkeypatch):
    def boom(_data):
        raise RuntimeError("bug")

    monkeypatch.setattr(router, "_gate_efficiency", boom)
    before = requests_count("local-fast", "fail-closed", "research")
    out, decision, spans, _proxy = route_under_proxy_span(router, LONG_BENIGN)
    assert out["model"] == "local-fast" and "fail-closed" in decision["reason"]
    assert requests_count("local-fast", "fail-closed", "research") == before + 1
    assert spans["gate.efficiency"].status.status_code == trace.StatusCode.ERROR


def test_requests_counter_and_privacy_histogram(router):
    before_local = requests_count("local-fast", "privacy", "research")
    before_sota = requests_count("sota-smart", "all-sota", "research")
    before_hist = REGISTRY.get_sample_value(
        "router_privacy_score_count", {"team": "research"}) or 0.0
    route(router, WITH_CARD)
    route(router, LONG_BENIGN)
    assert requests_count("local-fast", "privacy", "research") == before_local + 1
    assert requests_count("sota-smart", "all-sota", "research") == before_sota + 1
    assert REGISTRY.get_sample_value(
        "router_privacy_score_count", {"team": "research"}) == before_hist + 2


def test_team_label_is_bounded(router):
    before = requests_count("sota-smart", "all-sota", "other")
    route(router, LONG_BENIGN, team="some-unknown-team")
    assert requests_count("sota-smart", "all-sota", "other") == before + 1
    _out, decision = route(router, LONG_BENIGN, team=None)
    assert decision["team"] is None
    assert requests_count("sota-smart", "all-sota", "none") >= 1


def test_sota_budget_gauge(router):
    router.sota_token_budget = 1000
    kwargs = {"litellm_params": {"metadata": {"routing_decision": {"routed_to": "sota-smart"}}}}
    response = types.SimpleNamespace(model="x", usage=types.SimpleNamespace(total_tokens=42))
    asyncio.run(router.async_log_success_event(kwargs, response, None, None))
    assert REGISTRY.get_sample_value("router_sota_budget_used_tokens") == 42


def test_tracing_errors_never_change_the_decision(router, monkeypatch):
    expected = {case: route(router, text)[1] for case, text in
                (("local", WITH_CARD), ("sota", LONG_BENIGN))}

    class BrokenTracer:
        def start_span(self, *_args, **_kwargs):
            raise RuntimeError("tracer broken")

    monkeypatch.setattr(privacy_scoring.otel_trace, "get_tracer", lambda *_a, **_k: BrokenTracer())

    class BrokenMetric:
        def labels(self, **_kwargs):
            raise RuntimeError("metrics broken")

    monkeypatch.setattr(policy_hook_chain, "REQUESTS", BrokenMetric())
    monkeypatch.setattr(policy_hook_chain, "PRIVACY_SCORE", BrokenMetric())
    for case, text in (("local", WITH_CARD), ("sota", LONG_BENIGN)):
        out, decision = route(router, text)
        assert out["model"] == expected[case]["routed_to"]
        for key in ("routed_to", "decided_by", "chain", "reason", "team"):
            assert decision[key] == expected[case][key]
        assert "trace_id" not in decision


def test_log_line_has_the_trace_id(router, capsys):
    route(router, LONG_BENIGN)
    line = [x for x in capsys.readouterr().out.splitlines()
            if x.startswith("[policy-router] {")][-1]
    decision = ast.literal_eval(line.split("[policy-router] ", 1)[1])
    keys = {"policy", "requested", "routed_to", "decided_by", "chain", "reason", "team"}
    assert keys <= set(decision)
    assert len(decision["trace_id"]) == 32


def test_metrics_server_off_and_bind_error(monkeypatch):
    monkeypatch.delattr(policy_hook_chain.prometheus_client, "_policy_router_metrics_port",
                        raising=False)
    monkeypatch.setenv("ROUTER_METRICS_PORT", "0")
    assert policy_hook_chain._start_metrics_server() is None
    with socket.socket() as busy:
        busy.bind(("0.0.0.0", 0))
        busy.listen()
        monkeypatch.setenv("ROUTER_METRICS_PORT", str(busy.getsockname()[1]))
        assert policy_hook_chain._start_metrics_server() is None  # logged, not raised


def test_metrics_server_serves_the_router_metrics(monkeypatch, router):
    monkeypatch.delattr(policy_hook_chain.prometheus_client, "_policy_router_metrics_port",
                        raising=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("ROUTER_METRICS_PORT", str(port))
    assert policy_hook_chain._start_metrics_server() == port
    assert policy_hook_chain._start_metrics_server() == port  # once per process
    route(router, LONG_BENIGN)
    import urllib.request
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    assert "router_requests_total{" in body and "router_privacy_score_bucket" in body


NO_PACKAGES = r"""
import asyncio, sys, types
# The LiteLLM image has both packages; this run proves the hook works without them.
sys.modules["opentelemetry"] = None
sys.modules["prometheus_client"] = None
litellm = types.ModuleType("litellm")
integrations = types.ModuleType("litellm.integrations")
custom_logger = types.ModuleType("litellm.integrations.custom_logger")
class CustomLogger:
    def __init__(self, *a, **k):
        pass
custom_logger.CustomLogger = CustomLogger
sys.modules.update({"litellm": litellm, "litellm.integrations": integrations,
                    "litellm.integrations.custom_logger": custom_logger})
import policy_hook_chain as h
assert h.prometheus_client is None and h.REQUESTS is None
import privacy_scoring
assert privacy_scoring.otel_trace is None
r = h.ChainRouter()
r.scorer._lang_detector = lambda _t: ("en", 0.99)
async def no_entities(text, lang, timeout, entities=None):
    return []
r.scorer._query_presidio = no_entities
for text, target in ((sys.argv[1], "sota-smart"), (sys.argv[1] + sys.argv[2], "local-fast")):
    data = {"model": "auto", "messages": [{"role": "user", "content": text}],
            "metadata": {"headers": {"x-team": "research"}}}
    out = asyncio.run(r.async_pre_call_hook(None, None, data, "acompletion"))
    decision = out["metadata"]["routing_decision"]
    assert out["model"] == target, decision
    assert "trace_id" not in decision
print("OK without opentelemetry and prometheus_client")
"""


def test_hook_works_without_the_packages():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "litellm"), ROUTER_METRICS_PORT="9091")
    result = subprocess.run(
        [sys.executable, "-c", NO_PACKAGES, LONG_BENIGN, " Customer card 4111 1111 1111 1111."],
        env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK without opentelemetry and prometheus_client" in result.stdout
