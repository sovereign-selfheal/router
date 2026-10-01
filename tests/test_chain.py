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
    router.sota_token_budget = 1000
    kwargs = {"litellm_params": {"metadata": {"routing_decision": {"routed_to": "sota-smart"}}}}
    response = types.SimpleNamespace(model="x", usage=types.SimpleNamespace(total_tokens=42))
    asyncio.run(router.async_log_success_event(kwargs, response, None, None))
    assert router._sota_tokens_used == 42


def test_sota_budget_zero_counts_nothing_and_never_caps(router):
    router.sota_token_budget, router._sota_tokens_used = 0, 10**9
    kwargs = {"litellm_params": {"metadata": {"routing_decision": {"routed_to": "sota-smart"}}}}
    response = types.SimpleNamespace(model="x", usage=types.SimpleNamespace(total_tokens=42))
    asyncio.run(router.async_log_success_event(kwargs, response, None, None))
    assert router._sota_tokens_used == 10**9
    out, decision = route(router, LONG_BENIGN)
    assert out["model"] == "sota-smart"
    assert "SOTA budget" not in decision["reason"]


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


# ------------------------------------------------- SOTA size cap (v0.8.0)
def agent_request(tool_output, tools=None):
    """An agent turn: a system prompt, the question, a tool call and its output."""
    data = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "You are a self-heal agent."},
            {"role": "user", "content": LONG_BENIGN},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_logs", "arguments": '{"pod": "api-7"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": tool_output},
        ],
        "metadata": {"headers": {"x-team": "research"}},
    }
    if tools is not None:
        data["tools"] = tools
    return data


def route_data(router, data):
    out = asyncio.run(router.async_pre_call_hook(None, None, data, "acompletion"))
    return out, out["metadata"]["routing_decision"]


def test_size_cap_unset_keeps_the_old_behaviour(router):
    assert router.sota_max_prompt_chars == 0
    out, decision = route_data(router, agent_request("log line\n" * 5000))
    assert out["model"] == "sota-smart"
    assert decision["sota_cap"] is None
    assert decision["prompt_chars"] > 40000


def test_size_cap_below_goes_on_to_privacy(router):
    router.sota_max_prompt_chars = 100000
    out, decision = route_data(router, agent_request("log line\n" * 1000))
    assert out["model"] == "sota-smart"
    assert decision["decided_by"] == "all-sota"
    assert decision["sota_cap"] == 100000
    assert router.scorer._query_presidio.calls  # the privacy gate ran


def test_size_cap_above_routes_local_without_detectors(router):
    router.sota_max_prompt_chars = 5000
    out, decision = route_data(router, agent_request("log line\n" * 1000))
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "efficiency"
    assert "SOTA context limit" in decision["reason"]
    assert decision["prompt_chars"] > 5000
    assert router.scorer._query_presidio.calls == []  # no detector was called
    assert "ner_timeout_s" not in decision


def test_size_cap_counts_the_tool_definitions(router):
    data = agent_request("short output")
    size = router._prompt_chars(data)
    tools = [{"type": "function", "function": {"name": "get_logs", "description": "x" * 500}}]
    with_tools = agent_request("short output", tools=tools)
    assert router._prompt_chars(with_tools) > size + 500
    router.sota_max_prompt_chars = size + 100
    _out, decision = route_data(router, data)
    assert decision["routed_to"] == "sota-smart"
    _out, decision = route_data(router, with_tools)
    assert decision["routed_to"] == "local-fast"
    assert "SOTA context limit" in decision["reason"]


def test_size_cap_counting_error_fails_closed(router, monkeypatch):
    def boom(_data):
        raise RuntimeError("bad payload")

    router.sota_max_prompt_chars = 100000
    monkeypatch.setattr(router, "_prompt_chars", boom)
    out, decision = route_data(router, agent_request("log line"))
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "efficiency"
    assert "count error" in decision["reason"]
    assert decision["prompt_chars"] is None


def test_log_line_has_the_size_and_timeout_fields(router, capsys):
    route(router, LONG_BENIGN)
    lines = capsys.readouterr().out.splitlines()
    line = [x for x in lines if x.startswith("[policy-router] {")][-1]
    decision = ast.literal_eval(line.split("[policy-router] ", 1)[1])
    assert decision["prompt_chars"] == len(LONG_BENIGN)
    assert decision["sota_cap"] is None
    assert decision["ner_timeout_s"] == 3.0
    assert decision["c2_timeout_s"] == 8.0


def test_log_line_shows_the_effective_timeouts(router):
    router.scorer._ner["timeout_per_1k_chars"] = 0.05
    router.scorer._ner["timeout_max_seconds"] = 10.0
    _out, decision = route_data(router, agent_request("x" * 100000))
    assert 5.0 < decision["ner_timeout_s"] < 5.5
    assert decision["c2_timeout_s"] == 8.0
