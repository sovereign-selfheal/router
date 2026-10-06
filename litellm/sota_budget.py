"""SOTA token budget per tier (v0.12.0): when a tier used its SOTA tokens, it stays LOCAL.

The tier is the `x-team` header that the gateway sets from the API key (label `maas-group`,
the client cannot change it): one tier per agent or application. Each tier can have a budget of
SOTA tokens per time window (`efficiency.sota_budget.tiers` in chain.yaml, or the env
SOTA_BUDGET_TIERS as JSON). When the tokens of the last window reach the budget, the efficiency
gate keeps the request LOCAL: the agent goes on with the local model, instead of the 429 of the
gateway, which counts all the tokens and stays as the ceiling. A tier without a budget is not
limited here.

The counters live in Redis, so both LiteLLM pods share them and a pod restart keeps them. The
window slides: one key per fixed window (`sota-budget:<tier>:<window index>`, INCRBY with a
TTL of two windows), and the tokens used are the current window plus the share of the previous
one that still falls in the last `window_s` seconds. A request is checked before it is sent and
counted when its answer arrives, so parallel requests can go a little over the budget.

FAIL-OPEN: when Redis cannot be reached, the budget does not apply (normal routing), with a
log line and the metric router_sota_budget_store_errors_total. The budget controls cost, not
privacy: the gates still run.

No LiteLLM import: testable alone, like namespace_policy.
"""

import json
import os
import time

try:  # redis-py is in the LiteLLM image; optional so import never breaks the proxy
    import redis.asyncio as aioredis
except Exception:  # pragma: no cover - defensive
    aioredis = None

DEFAULT_WINDOW_S = 300
KEY_PREFIX = "sota-budget"
STORE_TIMEOUT_S = 0.1


class MemoryStore:
    """In-process store with the interface of RedisStore (tests; one pod only)."""

    def __init__(self):
        self.data = {}

    async def get_many(self, keys):
        return [int(self.data.get(k, 0)) for k in keys]

    async def incr(self, key, amount, ttl_s):
        self.data[key] = int(self.data.get(key, 0)) + int(amount)


class RedisStore:
    """The counters in Redis. Short timeouts: the check is on the path of every request."""

    def __init__(self, url, password=None, timeout_s=STORE_TIMEOUT_S):
        if aioredis is None:
            raise RuntimeError("redis-py unavailable")
        self._client = aioredis.from_url(
            url, password=password or None, decode_responses=True,
            socket_timeout=timeout_s, socket_connect_timeout=timeout_s,
        )

    async def get_many(self, keys):
        values = await self._client.mget(keys)
        return [int(v or 0) for v in values]

    async def incr(self, key, amount, ttl_s):
        pipe = self._client.pipeline(transaction=False)
        pipe.incrby(key, int(amount))
        pipe.expire(key, int(ttl_s))
        await pipe.execute()


def _tiers_from(value):
    """{tier: tokens} with lowercase names and positive integers; anything else is ignored."""
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else {}
    out = {}
    for name, tokens in (value or {}).items():
        try:
            tokens = int(tokens)
        except (TypeError, ValueError):
            continue
        if tokens > 0 and str(name).strip():
            out[str(name).strip().lower()] = tokens
    return out


class SotaBudget:
    """Policy key `efficiency.sota_budget` of chain.yaml, with the env overrides."""

    def __init__(self, cfg, store=None, clock=time.time):
        cfg = cfg or {}
        env = os.environ.get
        tiers = env("SOTA_BUDGET_TIERS")
        self.tiers = _tiers_from(tiers if tiers not in (None, "") else cfg.get("tiers"))
        window = env("SOTA_BUDGET_WINDOW_S") or cfg.get("window_s") or DEFAULT_WINDOW_S
        self.window_s = max(1, int(float(window)))
        self.clock = clock
        self.store = store
        self.error = None
        if self.tiers and self.store is None:
            url = env("SOTA_BUDGET_REDIS_URL") or cfg.get("redis_url") or ""
            if not url:
                self.error = "no store (SOTA_BUDGET_REDIS_URL is empty)"
            else:
                try:
                    self.store = RedisStore(url, password=env("SOTA_BUDGET_REDIS_PASSWORD"))
                except Exception as exc:
                    self.error = f"store error: {type(exc).__name__}: {exc}"
        self.enabled = bool(self.tiers) and self.store is not None
        if self.tiers:
            state = "on" if self.enabled else f"OFF ({self.error})"
            print(f"[policy-router] SOTA budget {state}: {self.tiers} tokens per "
                  f"{self.window_label()}", flush=True)

    def limit(self, tier):
        """The budget of the tier, or None when the tier is not limited (or the budget is off)."""
        if not self.enabled or not tier:
            return None
        return self.tiers.get(str(tier).lower())

    def window_label(self):
        w = self.window_s
        if w % 3600 == 0:
            return f"{w // 3600}h"
        return f"{w // 60}m" if w % 60 == 0 else f"{w}s"

    def _keys(self, tier, now):
        index = int(now // self.window_s)
        return f"{KEY_PREFIX}:{tier}:{index}", f"{KEY_PREFIX}:{tier}:{index - 1}"

    async def used(self, tier):
        """SOTA tokens of the tier in the last window (sliding)."""
        now = self.clock()
        current, previous = self._keys(str(tier).lower(), now)
        cur, prev = await self.store.get_many([current, previous])
        elapsed = (now % self.window_s) / self.window_s
        return cur + int(prev * (1.0 - elapsed))

    async def add(self, tier, tokens):
        """Count the SOTA tokens of one answer in the current window."""
        current, _ = self._keys(str(tier).lower(), self.clock())
        await self.store.incr(current, tokens, ttl_s=2 * self.window_s + 60)
