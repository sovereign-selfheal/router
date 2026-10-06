"""SOTA budget per tier (v0.12.0): a tier that used its SOTA tokens stays LOCAL."""

import asyncio
import types

import policy_hook_chain
import pytest
import sota_budget
from conftest import FakePresidio
from namespace_policy import NamespacePolicy
from prometheus_client import REGISTRY
from sota_budget import MemoryStore, SotaBudget

LONG_BENIGN = ("Analyze the trade-offs between event sourcing and a classic CRUD model for an "
               "order management system, with a short example.")


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class BrokenStore:
    async def get_many(self, keys):
        raise ConnectionError("redis down")

    async def incr(self, key, amount, ttl_s):
        raise ConnectionError("redis down")


def make_router(tiers=None, store=None, clock=None, window_s=300):
    r = policy_hook_chain.ChainRouter()
    r.scorer._lang_detector = lambda _text: ("en", 0.99)
    r.scorer._query_presidio = FakePresidio()
    r.budget = SotaBudget({"tiers": tiers or {"research": 100}, "window_s": window_s},
                          store=store or MemoryStore(), clock=clock or Clock())
    r._known_teams |= set(r.budget.tiers)
    return r


def route(router, text, team="research"):
    data = {"model": "auto", "messages": [{"role": "user", "content": text}],
            "metadata": {"headers": {"x-team": team}}}
    out = asyncio.run(router.async_pre_call_hook(None, None, data, "acompletion"))
    return out, (out.get("metadata") or {}).get("routing_decision")


def answer(router, team="research", tokens=150, group="sota-smart", routed_to="sota-smart"):
    """Call the success event like LiteLLM does after an answer."""
    kwargs = {
        "litellm_params": {"metadata": {"routing_decision": {"routed_to": routed_to,
                                                             "team": team}}},
        "standard_logging_object": {"model_group": group},
    }
    response = types.SimpleNamespace(model="x", usage=types.SimpleNamespace(total_tokens=tokens))
    asyncio.run(router.async_log_success_event(kwargs, response, None, None))


def sample(name, labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ------------------------------------------------------------------ the store and the window
def test_budget_is_off_without_tiers():
    b = SotaBudget({"window_s": 300}, store=MemoryStore())
    assert b.enabled is False
    assert b.limit("research") is None


def test_budget_without_store_url_is_off(monkeypatch, capsys):
    monkeypatch.delenv("SOTA_BUDGET_REDIS_URL", raising=False)
    b = SotaBudget({"tiers": {"agents": 1000}})
    assert b.enabled is False
    assert "SOTA budget OFF (no store" in capsys.readouterr().out


def test_env_overrides_the_policy(monkeypatch):
    monkeypatch.setenv("SOTA_BUDGET_TIERS", '{"Agents": 1500, "bad": "x", "zero": 0}')
    monkeypatch.setenv("SOTA_BUDGET_WINDOW_S", "60")
    b = SotaBudget({"tiers": {"research": 5}, "window_s": 300}, store=MemoryStore())
    assert b.tiers == {"agents": 1500}
    assert (b.window_s, b.window_label()) == (60, "1m")
    assert b.limit("AGENTS") == 1500 and b.limit("research") is None


def test_used_tokens_slide_with_the_window():
    clock = Clock(now=3000.0)  # start of a 300 s window
    b = SotaBudget({"tiers": {"agents": 1000}, "window_s": 300}, store=MemoryStore(), clock=clock)
    asyncio.run(b.add("agents", 600))
    assert asyncio.run(b.used("agents")) == 600
    clock.now += 300 + 150  # half of the next window: half of the previous one still counts
    assert asyncio.run(b.used("agents")) == 300
    clock.now += 300  # two windows later: nothing left
    assert asyncio.run(b.used("agents")) == 0


def test_redis_store_uses_one_key_per_window(monkeypatch):
    calls = []

    class Pipe:
        def incrby(self, key, amount):
            calls.append(("incrby", key, amount))

        def expire(self, key, ttl):
            calls.append(("expire", key, ttl))

        async def execute(self):
            return [1, True]

    class Client:
        async def mget(self, keys):
            calls.append(("mget", tuple(keys)))
            return ["7", None]

        def pipeline(self, transaction):
            return Pipe()

    def from_url(url, **kw):
        calls.append(("url", url, kw))
        return Client()

    fake = types.SimpleNamespace(from_url=from_url)
    monkeypatch.setattr(sota_budget, "aioredis", fake)
    monkeypatch.setenv("SOTA_BUDGET_REDIS_URL", "redis://litellm-redis:6379/0")
    monkeypatch.setenv("SOTA_BUDGET_REDIS_PASSWORD", "pw")
    b = SotaBudget({"tiers": {"agents": 10}, "window_s": 300}, clock=Clock(now=3000.0))
    assert b.enabled is True
    assert calls[0][1] == "redis://litellm-redis:6379/0" and calls[0][2]["password"] == "pw"
    assert calls[0][2]["socket_timeout"] == sota_budget.STORE_TIMEOUT_S
    asyncio.run(b.add("agents", 5))
    assert ("incrby", "sota-budget:agents:10", 5) in calls
    assert ("expire", "sota-budget:agents:10", 660) in calls
    assert asyncio.run(b.used("agents")) == 7
    assert ("mget", ("sota-budget:agents:10", "sota-budget:agents:9")) in calls


# ------------------------------------------------------------------ routing
def test_tier_under_budget_follows_the_gates():
    r = make_router()
    out, decision = route(r, LONG_BENIGN)
    assert out["model"] == "sota-smart"
    assert decision["decided_by"] == "all-sota"
    assert (decision["sota_budget_used"], decision["sota_budget_limit"]) == (0, 100)


def test_tier_over_budget_stays_local():
    r = make_router()
    answer(r, tokens=150)
    out, decision = route(r, LONG_BENIGN)
    assert out["model"] == "local-fast"
    assert decision["decided_by"] == "efficiency"
    assert decision["reason"] == ("efficiency: SOTA budget of tier research used "
                                  "(150/100 tokens in 5m) -> LOCAL")
    assert r.scorer._query_presidio.calls == []  # no privacy detector ran


def test_other_tiers_are_not_limited():
    r = make_router()
    answer(r, tokens=150)
    out, decision = route(r, LONG_BENIGN, team="agents")
    assert out["model"] == "sota-smart"
    assert "sota_budget_used" not in decision


def test_restricted_namespace_does_not_read_the_budget():
    calls = []

    class Store(MemoryStore):
        async def get_many(self, keys):
            calls.append(keys)
            return await super().get_many(keys)

    r = make_router(store=Store())
    r.namespaces = NamespacePolicy({"scan": True}, fetch=lambda: {"payments": "restricted"},
                                   start=False)
    r.namespaces.labels.refresh()
    _out, decision = route(r, LONG_BENIGN + ' up{namespace="payments"}')
    assert decision["decided_by"] == "namespace"
    assert calls == []


def test_store_error_fails_open():
    before = sample("router_sota_budget_store_errors_total", {"op": "read"})
    r = make_router(store=BrokenStore())
    out, decision = route(r, LONG_BENIGN)
    assert out["model"] == "sota-smart"  # cost control only: the gates decided
    assert decision["sota_budget_used"] is None
    assert decision["sota_budget_error"].startswith("ConnectionError")
    assert sample("router_sota_budget_store_errors_total", {"op": "read"}) == before + 1
    answer(r, tokens=10)  # the write error is logged and counted, never raised
    assert sample("router_sota_budget_store_errors_total", {"op": "write"}) >= 1


# ------------------------------------------------------------------ what counts as SOTA
def test_fallback_to_local_is_not_counted():
    r = make_router()
    answer(r, tokens=150, group="local-fast", routed_to="sota-smart")
    assert asyncio.run(r.budget.used("research")) == 0


def test_local_only_mode_is_not_counted(monkeypatch):
    monkeypatch.setenv("SOTA_ENABLED", "0")
    r = make_router()
    answer(r, tokens=150)
    assert asyncio.run(r.budget.used("research")) == 0


def test_sota_tokens_metric_per_tier():
    r = make_router(tiers={"budget-tier": 1000})
    before = sample("router_sota_tokens_total", {"team": "budget-tier"})
    answer(r, team="budget-tier", tokens=42)
    assert sample("router_sota_tokens_total", {"team": "budget-tier"}) == before + 42
    assert asyncio.run(r.budget.used("budget-tier")) == 42
    route(r, LONG_BENIGN, team="budget-tier")
    assert sample("router_sota_budget_window_tokens", {"team": "budget-tier"}) == 42


def test_limit_gauge_per_tier(monkeypatch):
    monkeypatch.setenv("SOTA_BUDGET_TIERS", '{"gauge-tier": 2500}')
    monkeypatch.setenv("SOTA_BUDGET_REDIS_URL", "")
    r = policy_hook_chain.ChainRouter()
    assert r.budget.tiers == {"gauge-tier": 2500}
    assert sample("router_sota_budget_limit_tokens", {"team": "gauge-tier"}) == 2500


@pytest.mark.parametrize("group", [None, ""])
def test_without_model_group_the_decision_decides(group):
    r = make_router()
    kwargs = {"litellm_params": {"metadata": {"routing_decision": {"routed_to": "sota-smart",
                                                                   "team": "research"}}}}
    if group is not None:
        kwargs["standard_logging_object"] = {"model_group": group}
    response = types.SimpleNamespace(model="x", usage=types.SimpleNamespace(total_tokens=9))
    asyncio.run(r.async_log_success_event(kwargs, response, None, None))
    assert asyncio.run(r.budget.used("research")) == 9
