"""PrivacyScorer: detectors A (rules), B (lexicons), C1 (NER), C2 (classifier), aggregation."""

import asyncio
import copy

import httpx
import pytest


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
