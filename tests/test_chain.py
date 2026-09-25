"""policy_hook_chain: gate order, tiering, fail-closed and the [policy-router] log line."""

import ast
import asyncio
import types

import policy_hook_chain
import pytest
from conftest import FakePresidio

LONG_BENIGN = ("Analyze the trade-offs between event sourcing and a classic CRUD model for an "
               "order management system, with a short example.")


@pytest.fixture
def router():
    """A fresh ChainRouter on the test policies, with a fake Presidio and English text."""
    r = policy_hook_chain.ChainRouter()
    r.scorer._lang_detector = lambda _text: ("en", 0.99)
    r.scorer._query_presidio = FakePresidio()
    return r


def route(router, text, team="research", call_type="acompletion"):
    data = {"model": "auto", "messages": [{"role": "user", "content": text}],
            "metadata": {"headers": {"x-team": team}} if team else {}}
    out = asyncio.run(router.async_pre_call_hook(None, None, data, call_type))
    return out, (out.get("metadata") or {}).get("routing_decision")


def test_short_question_stays_local(router):
    out, decision = route(router, "What is the capital of France?")
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "efficiency"


def test_long_benign_question_goes_to_sota(router):
    out, decision = route(router, LONG_BENIGN)
    assert out["model"] == "sota-smart"
    assert decision["decided_by"] == "all-sota"
    assert decision["team"] == "research"


def test_italian_keyword_marks_complex(router):
    _out, decision = route(router, "Confronta Kubernetes e le VM tradizionali per una banca.")
    assert decision["routed_to"] == "sota-smart"
    assert "keyword 'confront'" in decision["chain"][0]


def test_pii_keeps_complex_question_local(router):
    out, decision = route(router, LONG_BENIGN + " Customer card 4111 1111 1111 1111.")
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "privacy"


def test_legal_tier_never_reaches_sota(router):
    out, decision = route(router, LONG_BENIGN, team="legal")
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "tiering"


def test_sota_budget_exhausted_routes_local(router):
    router.sota_token_budget, router._sota_tokens_used = 100, 100
    _out, decision = route(router, LONG_BENIGN)
    assert decision["routed_to"] == "local-fast"
    assert "SOTA budget exhausted" in decision["reason"]


def test_sota_usage_is_counted(router):
    kwargs = {"litellm_params": {"metadata": {"routing_decision": {"routed_to": "sota-smart"}}}}
    response = types.SimpleNamespace(model="x", usage=types.SimpleNamespace(total_tokens=42))
    asyncio.run(router.async_log_success_event(kwargs, response, None, None))
    assert router._sota_tokens_used == 42


def test_unexpected_error_fails_closed(router, monkeypatch):
    def boom(_data):
        raise RuntimeError("bug")

    monkeypatch.setattr(router, "_gate_efficiency", boom)
    out, decision = route(router, LONG_BENIGN)
    assert out["model"] == "local-fast"
    assert "fail-closed" in decision["reason"]


def test_other_call_types_are_untouched(router):
    out, decision = route(router, LONG_BENIGN, call_type="embeddings")
    assert out["model"] == "auto"
    assert decision is None


def test_log_line_contract(router, capsys):
    """The e2e harness and the demo video parse this line: keep its shape."""
    route(router, LONG_BENIGN)
    lines = capsys.readouterr().out.splitlines()
    line = [x for x in lines if x.startswith("[policy-router] {")][-1]
    decision = ast.literal_eval(line.split("[policy-router] ", 1)[1])
    keys = {"policy", "requested", "routed_to", "decided_by", "chain", "reason", "team"}
    assert keys <= set(decision)
