"""Shared test setup: a fake LiteLLM module, the test policies and a fake Presidio.

The tests need no network, no LiteLLM and no fastText model.
"""

import os
import pathlib
import sys
import types

import pytest
import yaml

POLICY_DIR = pathlib.Path(__file__).parent / "policy"

# policy_hook_chain.py reads POLICY_DIR at import time and creates its handler.
os.environ["POLICY_DIR"] = str(POLICY_DIR)
# No metrics HTTP server in the tests (the tests read the registry directly).
os.environ["ROUTER_METRICS_PORT"] = "0"
# The C2 classifier must stay off unless a test turns it on.
for _var in ("CLASSIFIER_ENABLED", "CLASSIFIER_GRAY_LOW", "SOTA_SERVED_MATCH"):
    os.environ.pop(_var, None)

# Minimal stand-in for litellm.integrations.custom_logger.CustomLogger.
if "litellm" not in sys.modules:
    _litellm = types.ModuleType("litellm")
    _integrations = types.ModuleType("litellm.integrations")
    _custom_logger = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:  # noqa: D401 - test stub
        def __init__(self, *args, **kwargs):
            pass

    _custom_logger.CustomLogger = CustomLogger
    _litellm.integrations = _integrations
    _integrations.custom_logger = _custom_logger
    sys.modules.update({
        "litellm": _litellm,
        "litellm.integrations": _integrations,
        "litellm.integrations.custom_logger": _custom_logger,
    })

from privacy_scoring import PrivacyScorer  # noqa: E402  (after the stubs)


def load_policy(name):
    return yaml.safe_load((POLICY_DIR / f"{name}.yaml").read_text())


class FakePresidio:
    """Answers /analyze from a list of (substring, entity_type, score).

    Every occurrence of a substring in the text becomes one entity, with its real
    start and end offsets, like Presidio does. `calls` records the languages asked.
    """

    def __init__(self, entities=(), error=None):
        self.entities = list(entities)
        self.error = error
        self.calls = []

    async def __call__(self, text, lang, timeout, entities=None):
        self.calls.append(lang)
        if self.error:
            raise self.error
        found = []
        for needle, etype, score in self.entities:
            if entities and etype not in entities:
                continue
            start = text.find(needle)
            while start >= 0:
                found.append({"entity_type": etype, "start": start,
                              "end": start + len(needle), "score": score})
                start = text.find(needle, start + 1)
        return found


@pytest.fixture
def privacy_policy():
    return load_policy("privacy-plus")


@pytest.fixture
def legacy_policy(privacy_policy):
    """The policy without the keys added in v0.4.0: the behaviour before them."""
    for key in ("person_min_words", "context_entities"):
        privacy_policy["ner"].pop(key, None)
    return privacy_policy


@pytest.fixture
def make_scorer(privacy_policy):
    """Build a PrivacyScorer with a fake Presidio and a fixed language."""

    def _make(entities=(), lang=("en", 0.99), policy=None, error=None):
        scorer = PrivacyScorer(policy or privacy_policy, lang_detector=lambda _text: lang)
        scorer._query_presidio = scorer.fake_presidio = FakePresidio(entities, error)
        return scorer

    return _make
