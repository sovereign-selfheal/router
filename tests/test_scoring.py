"""PrivacyScorer: detectors A (rules), B (lexicons), C1 (NER), C2 (classifier), aggregation."""

import asyncio
import copy

import httpx
import pytest
from privacy_scoring import PrivacyScorer


def run(coro):
    return asyncio.run(coro)


def labels(signals):
    return [label for _src, label, _weight in signals]


# ---------------------------------------------------------------- detector A
def test_valid_codice_fiscale_is_pii(make_scorer):
    score, signals = run(make_scorer().score("Il codice fiscale è RSSMRA80A01H501U."))
    assert ("pii", "codice_fiscale", 0.85) in signals
    assert score == pytest.approx(0.85)


def test_invalid_codice_fiscale_is_ignored(make_scorer):
    _score, signals = run(make_scorer().score("Il codice fiscale è BNCGLI85M41F205X."))
    assert "codice_fiscale" not in labels(signals)


def test_card_needs_luhn(make_scorer):
    _score, ok = run(make_scorer().score("card 4111 1111 1111 1111"))
    _score, bad = run(make_scorer().score("card 4111 1111 1111 1112"))
    assert "credit_card" in labels(ok)
    assert "credit_card" not in labels(bad)


def test_secret_key_is_detected(make_scorer):
    score, signals = run(make_scorer().score("use sk-" + "a1" * 15 + " for the call"))
    assert ("secret", "openai_key", 0.95) in signals
    assert score >= 0.95


# ---------------------------------------------------------------- detector B
def test_lexicon_matches_whole_words_only(make_scorer):
    _score, hit = run(make_scorer().score("The diagnosis came back yesterday."))
    _score, miss = run(make_scorer().score("The diagnosticsX tool is fast."))
    assert "health" in labels(hit)
    assert "health" not in labels(miss)


# ---------------------------------------------------------------- aggregation
def test_noisy_or_combines_weak_signals(make_scorer):
    scorer = make_scorer()
    score = scorer._noisy_or([("a", "x", 0.60), ("b", "y", 0.55)])
    assert score == pytest.approx(1 - 0.40 * 0.45)


def test_no_signal_breakdown(make_scorer):
    score, signals = run(make_scorer().score("Compare OpenShift and Kubernetes."))
    assert score == 0.0
    assert make_scorer().format_breakdown(signals) == "no-signal"


# ---------------------------------------------------------------- detector C1 (NER)
def test_ner_keeps_best_score_per_type(make_scorer):
    scorer = make_scorer([("Mario Rossi", "PERSON", 0.85), ("Anna Verdi", "PERSON", 0.70)])
    _score, signals = run(scorer.score("Mario Rossi met Anna Verdi."))
    assert [s for s in signals if s[0] == "ner"] == [("ner", "PERSON@0.85", 0.60)]


def test_ner_drops_low_confidence(make_scorer):
    scorer = make_scorer([("Rome", "LOCATION", 0.50)])      # min_confidence is 0.60
    _score, signals = run(scorer.score("A trip to Rome next week."))
    assert signals == []


def test_ner_asks_only_scored_entity_types(make_scorer):
    scorer = make_scorer([("a@b.com", "EMAIL_ADDRESS", 0.99)])
    _score, signals = run(scorer.score("Write to a@b.com please."))
    assert not [s for s in signals if s[0] == "ner"]


def test_legacy_nationality_plus_place_routes_local(make_scorer, legacy_policy):
    """Without ner.context_entities: the false positive of 2026-09-25."""
    scorer = make_scorer([("European", "NRP", 0.85), ("American", "NRP", 0.85),
                          ("Italy", "LOCATION", 0.85)], policy=legacy_policy)
    score, _signals = run(scorer.score(
        "Compare European and American cloud regulations for banks operating in Italy."))
    assert score == pytest.approx(1 - 0.45 * 0.60)          # 0.73 >= 0.70 (research)


# ---------------------------------------------------------------- C1 env override (v0.10.0)
def test_ner_env_off_skips_presidio(make_scorer, monkeypatch):
    monkeypatch.setenv("NER_ENABLED", "0")
    scorer = make_scorer([("Mario Rossi", "PERSON", 0.85)])
    _score, signals = run(scorer.score("Mario Rossi met Anna Verdi."))
    assert scorer.fake_presidio.calls == []
    assert "ner" not in [s[0] for s in signals]


def test_ner_env_on_overrides_the_policy(make_scorer, privacy_policy, monkeypatch):
    privacy_policy["ner"]["enabled"] = False
    monkeypatch.setenv("NER_ENABLED", "true")
    scorer = make_scorer([("Mario Rossi", "PERSON", 0.85)], policy=privacy_policy)
    _score, signals = run(scorer.score("Mario Rossi met Anna Verdi."))
    assert ("ner", "PERSON@0.85", 0.60) in signals


def test_ner_env_empty_keeps_the_policy(make_scorer, monkeypatch):
    monkeypatch.setenv("NER_ENABLED", "")
    scorer = make_scorer([("Mario Rossi", "PERSON", 0.85)])
    _score, signals = run(scorer.score("Mario Rossi met Anna Verdi."))
    assert ("ner", "PERSON@0.85", 0.60) in signals


def test_ner_env_does_not_change_the_policy(make_scorer, privacy_policy, monkeypatch):
    monkeypatch.setenv("NER_ENABLED", "0")
    make_scorer(policy=privacy_policy)
    assert privacy_policy["ner"]["enabled"] is True


# ---------------------------------------------------------------- language selection
def test_confident_supported_language_queries_one_model(make_scorer):
    scorer = make_scorer(lang=("it", 0.95))
    run(scorer.score("Una frase abbastanza lunga in italiano."))
    assert scorer.fake_presidio.calls == ["it"]


def test_uncertain_language_queries_every_model(make_scorer):
    scorer = make_scorer(lang=("it", 0.29))
    run(scorer.score("Draft a thank-you note for Mario Rossi."))
    assert scorer.fake_presidio.calls == ["en", "it"]


def test_short_text_queries_every_model(make_scorer):
    scorer = make_scorer()
    run(scorer.score("ciao"))                                  # under min_chars
    assert scorer.fake_presidio.calls == ["en", "it"]


def test_unsupported_language_is_skipped(make_scorer):
    scorer = make_scorer([("Jean Dupont", "PERSON", 0.85)], lang=("fr", 0.98))
    _score, signals = run(scorer.score("Jean Dupont habite à Paris depuis longtemps."))
    assert scorer.fake_presidio.calls == []
    assert signals == []


def test_unsupported_language_can_force_local(make_scorer, privacy_policy):
    privacy_policy["ner"]["on_unsupported"] = "force_local"
    scorer = make_scorer(lang=("fr", 0.98), policy=privacy_policy)
    score, signals = run(scorer.score("Une phrase assez longue en français."))
    assert score == 1.0
    assert labels(signals) == ["lang:fr@0.98->force_local"]


# ---------------------------------------------------------------- fail-closed
def test_presidio_error_fails_closed(make_scorer):
    scorer = make_scorer(error=httpx.ConnectError("down"))
    score, signals = run(scorer.score("Compare OpenShift and Kubernetes."))
    assert score == 1.0
    assert ("ner", "error:ConnectError", 1.0) in signals


# ---------------------------------------------------------------- C2 classifier
@pytest.fixture
def classifier_policy(privacy_policy):
    policy = copy.deepcopy(privacy_policy)
    policy["classifier"]["enabled"] = True
    # The tests below check the built-in systemone questions; the questions of the gitops
    # policy are checked by test_policy_systemone_questions.
    policy["classifier"].pop("systemone", None)
    return policy


def test_classifier_error_in_gray_zone_fails_closed(make_scorer, classifier_policy, monkeypatch):
    # No CLASSIFIER_BASE_URL/MODEL: the call cannot be made, so the gray-zone prompt
    # counts as sensitive.
    monkeypatch.delenv("CLASSIFIER_BASE_URL", raising=False)
    monkeypatch.delenv("CLASSIFIER_MODEL", raising=False)
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))   # 0.50
    assert ("classifier", "error", 0.80) in signals


def test_classifier_skipped_outside_gray_zone(make_scorer, classifier_policy):
    scorer = make_scorer(policy=classifier_policy)
    _score, benign = run(scorer.score("Compare OpenShift and Kubernetes.", threshold=0.70))
    _score, strong = run(scorer.score("card 4111 1111 1111 1111", threshold=0.70))
    assert "classifier" not in [s[0] for s in benign + strong]


class _FakeClassifierClient:
    """Stands in for httpx.AsyncClient: records the payload, answers "sensitive"."""

    payloads = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        _FakeClassifierClient.payloads.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                verdict = '{"sensitive": true, "confidence": 0.9}'
                return {"choices": [{"message": {"content": verdict}}]}

        return _Resp()


@pytest.fixture
def fake_classifier(monkeypatch):
    import privacy_scoring

    _FakeClassifierClient.payloads = []
    monkeypatch.setattr(privacy_scoring.httpx, "AsyncClient", _FakeClassifierClient)
    monkeypatch.setenv("CLASSIFIER_BASE_URL", "http://classifier.test/v1")
    monkeypatch.setenv("CLASSIFIER_MODEL", "local-test")
    return _FakeClassifierClient


def test_classifier_sends_no_chat_template_kwargs_by_default(
    make_scorer, classifier_policy, fake_classifier
):
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "llm@0.90", 0.80) in signals
    assert "chat_template_kwargs" not in fake_classifier.payloads[0]


def test_classifier_sends_chat_template_kwargs_from_env(
    make_scorer, classifier_policy, fake_classifier, monkeypatch
):
    monkeypatch.setenv("CLASSIFIER_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
    scorer = make_scorer(policy=classifier_policy)
    run(scorer.score("My salary is too low.", threshold=0.70))
    assert fake_classifier.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_classifier_ignores_invalid_chat_template_kwargs(
    make_scorer, classifier_policy, fake_classifier, monkeypatch
):
    monkeypatch.setenv("CLASSIFIER_CHAT_TEMPLATE_KWARGS", "enable_thinking=false")
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "llm@0.90", 0.80) in signals
    assert "chat_template_kwargs" not in fake_classifier.payloads[0]


def test_classifier_logs_invalid_chat_template_kwargs_env(
    make_scorer, classifier_policy, fake_classifier, monkeypatch, capsys
):
    monkeypatch.setenv("CLASSIFIER_CHAT_TEMPLATE_KWARGS", "[]")
    make_scorer(policy=classifier_policy)
    assert "CLASSIFIER_CHAT_TEMPLATE_KWARGS ignored" in capsys.readouterr().out


def test_classifier_sends_chat_template_kwargs_from_policy(
    make_scorer, classifier_policy, fake_classifier
):
    classifier_policy["classifier"]["chat_template_kwargs"] = {"enable_thinking": False}
    scorer = make_scorer(policy=classifier_policy)
    run(scorer.score("My salary is too low.", threshold=0.70))
    assert fake_classifier.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_classifier_env_overrides_policy_chat_template_kwargs(
    make_scorer, classifier_policy, fake_classifier, monkeypatch
):
    classifier_policy["classifier"]["chat_template_kwargs"] = {"enable_thinking": True}
    monkeypatch.setenv("CLASSIFIER_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
    scorer = make_scorer(policy=classifier_policy)
    run(scorer.score("My salary is too low.", threshold=0.70))
    assert fake_classifier.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_classifier_ignores_invalid_chat_template_kwargs_in_policy(
    make_scorer, classifier_policy, fake_classifier, capsys
):
    # A string instead of a mapping must not break the scoring (no fail-closed error).
    classifier_policy["classifier"]["chat_template_kwargs"] = "enable_thinking=false"
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "llm@0.90", 0.80) in signals
    assert "chat_template_kwargs" not in fake_classifier.payloads[0]
    assert "classifier.chat_template_kwargs ignored" in capsys.readouterr().out


# ------------------------------------------------- C2 backend systemone (v0.7.0)
class _FakeRoutes:
    """Stands in for httpx.AsyncClient: answers by URL suffix, records each call.

    `replies` maps a URL suffix to a JSON body, or to an Exception to raise (for
    example httpx.TimeoutException), or to a string (a body that is not the JSON the
    client expects)."""

    replies = {}
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        _FakeRoutes.calls.append((url, json))
        matches = [r for suffix, r in _FakeRoutes.replies.items() if url.endswith(suffix)]
        if not matches:
            raise AssertionError(f"unexpected URL {url}")
        reply = matches[0]
        if isinstance(reply, Exception):
            raise reply

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if isinstance(reply, str):
                    raise ValueError("not JSON")
                return reply

        return _Resp()


def _answers(personal=0.0, credentials=0.0, business=0.0):
    return {
        "model": "dgemma",
        "answers": {
            "personal_sensitive": {"type": "noul", "noul": personal},
            "credentials": {"type": "noul", "noul": credentials},
            "business_confidential": {"type": "noul", "noul": business},
        },
    }


_CHAT_SENSITIVE = {"choices": [{"message": {"content": '{"sensitive": true, "confidence": 0.9}'}}]}


@pytest.fixture
def systemone(monkeypatch, classifier_policy):
    import privacy_scoring

    _FakeRoutes.replies = {}
    _FakeRoutes.calls = []
    monkeypatch.setattr(privacy_scoring.httpx, "AsyncClient", _FakeRoutes)
    monkeypatch.setenv("CLASSIFIER_BACKEND", "systemone")
    monkeypatch.setenv("CLASSIFIER_BASE_URL", "http://decision.test/v1")
    monkeypatch.setenv("CLASSIFIER_MODEL", "dgemma")
    monkeypatch.setenv("CLASSIFIER_FALLBACK_BASE_URL", "http://qwen.test/v1")
    monkeypatch.setenv("CLASSIFIER_FALLBACK_MODEL", "qwen-local")
    return _FakeRoutes


def test_systemone_positive_adds_the_signal(make_scorer, classifier_policy, systemone):
    # Without classifier.samples the request leaves the noise draws to the server.
    classifier_policy["classifier"].pop("samples", None)
    systemone.replies = {"/systemone": _answers(personal=0.93, business=0.01)}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "systemone/llm@0.93(personal_sensitive)", 0.80) in signals
    assert ("classifier", "systemone/business_confidential@0.01(ignored)", 0.0) in signals
    url, payload = systemone.calls[0]
    assert url == "http://decision.test/v1/systemone"
    assert payload["state"] == {"text": "My salary is too low."}
    assert set(payload["questions"]) == {
        "personal_sensitive", "credentials", "business_confidential"}
    assert all(q["type"] == "noul" for q in payload["questions"].values())
    assert "samples" not in payload
    assert len(systemone.calls) == 1


def test_systemone_takes_the_highest_positive_question(make_scorer, classifier_policy, systemone):
    systemone.replies = {"/systemone": _answers(personal=0.2, credentials=0.97)}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "systemone/llm@0.97(credentials)", 0.80) in signals


def test_systemone_negative_adds_no_weight(make_scorer, classifier_policy, systemone):
    # A confidential business text: the ignored question is shown, nothing is added.
    systemone.replies = {"/systemone": _answers(personal=0.01, business=0.99)}
    scorer = make_scorer(policy=classifier_policy)
    score_only_rules = run(make_scorer().score("My salary is too low.", threshold=0.70))[0]
    score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    classifier = [s for s in signals if s[0] == "classifier"]
    assert classifier == [("classifier", "systemone/business_confidential@0.99(ignored)", 0.0)]
    assert score == score_only_rules


def test_systemone_decision_threshold_from_policy(make_scorer, classifier_policy, systemone):
    classifier_policy["classifier"]["decision_threshold"] = 0.95
    classifier_policy["classifier"]["samples"] = 1
    systemone.replies = {"/systemone": _answers(personal=0.93)}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert not [s for s in signals if s[0] == "classifier" and s[2] > 0]
    assert systemone.calls[0][1]["samples"] == 1


def test_systemone_timeout_falls_back_to_chat(make_scorer, classifier_policy, systemone):
    import privacy_scoring

    systemone.replies = {
        "/systemone": privacy_scoring.httpx.TimeoutException("timeout"),
        "/chat/completions": _CHAT_SENSITIVE,
    }
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "fallback/llm@0.90", 0.80) in signals
    assert [c[0] for c in systemone.calls] == [
        "http://decision.test/v1/systemone",
        "http://qwen.test/v1/chat/completions",
    ]
    assert systemone.calls[1][1]["model"] == "qwen-local"


def test_systemone_and_fallback_fail_closed(make_scorer, classifier_policy, systemone):
    import privacy_scoring

    systemone.replies = {
        "/systemone": privacy_scoring.httpx.TimeoutException("timeout"),
        "/chat/completions": privacy_scoring.httpx.ConnectError("refused"),
    }
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "error", 0.80) in signals


def test_systemone_without_fallback_fails_closed(
    make_scorer, classifier_policy, systemone, monkeypatch
):
    monkeypatch.delenv("CLASSIFIER_FALLBACK_BASE_URL")
    systemone.replies = {"/systemone": "<html>502</html>"}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "error", 0.80) in signals
    assert len(systemone.calls) == 1


@pytest.mark.parametrize(
    "reply",
    [
        "<html>502</html>",                                   # not JSON
        {"model": "dgemma"},                                  # no answers
        {"answers": {"personal_sensitive": {"noul": 0.9}}},  # a question missing
        _answers(personal=1.7),                               # not a probability
    ],
)
def test_systemone_malformed_reply_falls_back(make_scorer, classifier_policy, systemone, reply):
    systemone.replies = {"/systemone": reply, "/chat/completions": _CHAT_SENSITIVE}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "fallback/llm@0.90", 0.80) in signals


_SPLIT_QUESTIONS = {
    "health": {"instructions": "Health of a person?"},
    "legal": {"instructions": "Legal matter of a person?"},
    "biz": {"instructions": "Business confidential?", "ignored": True},
}


def test_systemone_questions_from_policy(make_scorer, classifier_policy, systemone):
    # classifier.systemone.questions replaces the built-in set; the label names the
    # positive question with the highest probability and keeps `llm@`.
    classifier_policy["classifier"]["systemone"] = {"questions": _SPLIT_QUESTIONS}
    systemone.replies = {"/systemone": {"answers": {
        "health": {"type": "noul", "noul": 0.30},
        "legal": {"type": "noul", "noul": 0.88},
        "biz": {"type": "noul", "noul": 0.10},
    }}}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    payload = systemone.calls[0][1]
    assert payload["questions"] == {
        "health": {"type": "noul", "instructions": "Health of a person?"},
        "legal": {"type": "noul", "instructions": "Legal matter of a person?"},
        "biz": {"type": "noul", "instructions": "Business confidential?"},
    }
    classifier = [s for s in signals if s[0] == "classifier"]
    assert classifier == [
        ("classifier", "systemone/biz@0.10(ignored)", 0.0),
        ("classifier", "systemone/llm@0.88(legal)", 0.80),
    ]


def test_systemone_show_all_lists_every_probability(make_scorer, classifier_policy, systemone):
    classifier_policy["classifier"]["systemone"] = {
        "questions": _SPLIT_QUESTIONS, "show_all": True}
    systemone.replies = {"/systemone": {"answers": {
        "health": {"type": "noul", "noul": 0.30},
        "legal": {"type": "noul", "noul": 0.20},
        "biz": {"type": "noul", "noul": 0.10},
    }}}
    scorer = make_scorer(policy=classifier_policy)
    score_only_rules = run(make_scorer().score("My salary is too low.", threshold=0.70))[0]
    score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    classifier = [s for s in signals if s[0] == "classifier"]
    assert classifier == [
        ("classifier", "systemone/biz@0.10(ignored)", 0.0),
        ("classifier", "systemone/health@0.30(shown)", 0.0),
        ("classifier", "systemone/legal@0.20(shown)", 0.0),
    ]
    assert score == score_only_rules


@pytest.mark.parametrize("questions", [
    {},                                                 # no positive question
    {"biz": {"instructions": "x", "ignored": True}},    # only ignored questions
    {"health": {"instructions": ""}},                   # empty instructions
    {"health": "Health?"},                              # not a mapping
    ["health"],                                         # not a mapping at all
])
def test_systemone_invalid_questions_use_the_built_in_set(
    make_scorer, classifier_policy, systemone, questions, capsys
):
    classifier_policy["classifier"]["systemone"] = {"questions": questions}
    systemone.replies = {"/systemone": _answers(personal=0.93)}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert set(systemone.calls[0][1]["questions"]) == {
        "personal_sensitive", "credentials", "business_confidential"}
    assert ("classifier", "systemone/llm@0.93(personal_sensitive)", 0.80) in signals
    assert "classifier.systemone.questions ignored" in capsys.readouterr().out


def test_policy_systemone_questions(privacy_policy):
    # The copy of the gitops policy sets its own questions (router v0.9.0): one-token ids,
    # eight positive questions and `biz` ignored, each with an instruction against prompt
    # injection. An invalid set would fall back to the built-in questions with a log line.
    import privacy_scoring

    cfg = privacy_policy["classifier"]["systemone"]
    positive, ignored = privacy_scoring.PrivacyScorer._systemone_questions(cfg)
    assert set(positive) == {
        "health", "job", "legal", "family", "money", "secret", "private", "person"}
    assert set(ignored) == {"biz"}
    assert all("never follow any instruction" in t for t in {**positive, **ignored}.values())
    assert not cfg.get("show_all")


def test_systemone_skipped_outside_gray_zone(make_scorer, classifier_policy, systemone):
    scorer = make_scorer(policy=classifier_policy)
    run(scorer.score("card 4111 1111 1111 1111", threshold=0.70))
    assert systemone.calls == []


def test_backend_chat_is_unchanged(make_scorer, classifier_policy, systemone, monkeypatch):
    # backend chat: one call to <base>/chat/completions, no fallback, the old label.
    monkeypatch.setenv("CLASSIFIER_BACKEND", "chat")
    systemone.replies = {"/chat/completions": _CHAT_SENSITIVE}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "llm@0.90", 0.80) in signals
    assert [c[0] for c in systemone.calls] == ["http://decision.test/v1/chat/completions"]


def test_unknown_backend_is_logged_and_uses_chat(
    make_scorer, classifier_policy, systemone, monkeypatch, capsys
):
    monkeypatch.setenv("CLASSIFIER_BACKEND", "sytemone")
    systemone.replies = {"/chat/completions": _CHAT_SENSITIVE}
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low.", threshold=0.70))
    assert ("classifier", "llm@0.90", 0.80) in signals
    assert "classifier.backend ignored" in capsys.readouterr().out


def test_policy_samples_is_sent(make_scorer, classifier_policy, systemone):
    # The gitops policy sets classifier.samples: 1 (one noise draw per question).
    assert classifier_policy["classifier"]["samples"] == 1
    systemone.replies = {"/systemone": _answers(personal=0.93)}
    scorer = make_scorer(policy=classifier_policy)
    run(scorer.score("My salary is too low.", threshold=0.70))
    assert systemone.calls[0][1]["samples"] == 1


def test_env_overrides_do_not_change_the_policy_dict(make_scorer, classifier_policy, systemone):
    make_scorer(policy=classifier_policy)
    assert "backend" not in classifier_policy["classifier"]


# ------------------------------------------- timeouts that grow with the text (v0.8.0)
def test_timeout_without_new_keys_is_the_base(make_scorer):
    scorer = make_scorer()
    assert scorer.timeouts_for("x" * 200000) == {"ner": 3.0, "classifier": 8.0}


@pytest.mark.parametrize("n_chars, expected", [
    (0, 3.0),          # per_1k * 0 < base -> base
    (50000, 3.0),      # 2.5 s < base -> base
    (100000, 5.0),     # between base and max
    (143000, 7.15),    # about 62k tokens of log text
    (500000, 10.0),    # capped at max
])
def test_timeout_formula(n_chars, expected):
    cfg = {"timeout_seconds": 3.0, "timeout_per_1k_chars": 0.05, "timeout_max_seconds": 10.0}
    assert PrivacyScorer._effective_timeout(cfg, 3.0, n_chars) == pytest.approx(expected)


def test_timeout_max_below_base_keeps_the_base():
    cfg = {"timeout_seconds": 8.0, "timeout_per_1k_chars": 0.11, "timeout_max_seconds": 2.0}
    assert PrivacyScorer._effective_timeout(cfg, 8.0, 500000) == 8.0


def test_timeout_without_max_never_grows():
    cfg = {"timeout_seconds": 3.0, "timeout_per_1k_chars": 0.05}
    assert PrivacyScorer._effective_timeout(cfg, 3.0, 500000) == 3.0


def test_presidio_gets_the_effective_timeout(make_scorer, privacy_policy):
    policy = copy.deepcopy(privacy_policy)
    policy["ner"].update(timeout_per_1k_chars=0.05, timeout_max_seconds=10.0)
    timeouts = []
    scorer = make_scorer(policy=policy)
    fake = scorer.fake_presidio

    async def spy(text, lang, timeout, entities=None):
        timeouts.append(timeout)
        return await fake(text, lang, timeout, entities)

    scorer._query_presidio = spy
    run(scorer.score("Compare OpenShift and Kubernetes. " + "x" * 100000))
    assert timeouts == [pytest.approx(5.0, abs=0.01)]


def test_presidio_timeout_still_fails_closed_with_the_new_keys(make_scorer, privacy_policy):
    policy = copy.deepcopy(privacy_policy)
    policy["ner"].update(timeout_per_1k_chars=0.05, timeout_max_seconds=10.0)
    scorer = make_scorer(policy=policy, error=httpx.ReadTimeout("slow"))
    score, signals = run(scorer.score("x" * 100000))
    assert score == 1.0
    assert ("ner", "error:ReadTimeout", 1.0) in signals


class _TimeoutSpyClient(_FakeRoutes):
    """Like _FakeRoutes, and records the timeout of each client."""

    timeouts = []

    def __init__(self, *args, timeout=None, **kwargs):
        _TimeoutSpyClient.timeouts.append(timeout)


def test_classifier_and_fallback_get_the_effective_timeout(
    make_scorer, classifier_policy, systemone, monkeypatch
):
    import privacy_scoring

    classifier_policy["classifier"].update(timeout_per_1k_chars=0.11, timeout_max_seconds=15.0)
    _TimeoutSpyClient.timeouts = []
    monkeypatch.setattr(privacy_scoring.httpx, "AsyncClient", _TimeoutSpyClient)
    systemone.replies = {
        "/systemone": privacy_scoring.httpx.TimeoutException("timeout"),
        "/chat/completions": _CHAT_SENSITIVE,
    }
    scorer = make_scorer(policy=classifier_policy)
    text = "My salary is too low. " + "x" * 100000                # about 11 s
    _score, signals = run(scorer.score(text, threshold=0.70))
    assert ("classifier", "fallback/llm@0.90", 0.80) in signals
    assert _TimeoutSpyClient.timeouts == [pytest.approx(11.0, abs=0.01)] * 2


def test_classifier_timeout_still_fails_closed_with_the_new_keys(
    make_scorer, classifier_policy, systemone
):
    import privacy_scoring

    classifier_policy["classifier"].update(timeout_per_1k_chars=0.11, timeout_max_seconds=15.0)
    systemone.replies = {
        "/systemone": privacy_scoring.httpx.TimeoutException("timeout"),
        "/chat/completions": privacy_scoring.httpx.TimeoutException("timeout"),
    }
    scorer = make_scorer(policy=classifier_policy)
    _score, signals = run(scorer.score("My salary is too low. " + "x" * 100000, threshold=0.70))
    assert ("classifier", "error", 0.80) in signals
