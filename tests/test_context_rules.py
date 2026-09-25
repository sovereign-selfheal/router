"""ner.person_min_words (D) and ner.context_entities (E).

The cases come from the e2e test of 2026-09-25: generic English questions of the demo
stayed LOCAL because Presidio read nationalities, places and product names as personal
data. The research team threshold is 0.70.
"""

import asyncio
import copy

import pytest

RESEARCH = 0.70


@pytest.fixture
def ed_policy(legacy_policy):
    policy = copy.deepcopy(legacy_policy)
    policy["ner"]["person_min_words"] = 2
    policy["ner"]["context_entities"] = ["NRP", "LOCATION"]
    return policy


def score(scorer, text):
    return asyncio.run(scorer.score(text, threshold=RESEARCH))


def breakdown(scorer, signals):
    return scorer.format_breakdown(signals)


# ---------------------------------------------------------------- defaults: behaviour unchanged
def test_defaults_keep_single_word_person(make_scorer, legacy_policy):
    scorer = make_scorer([("Kafka", "PERSON", 0.85)], policy=legacy_policy)
    value, _signals = score(scorer, "Compare Kafka and Pulsar for event streaming at scale.")
    assert value == pytest.approx(0.60)


def test_defaults_keep_context_entities(make_scorer, legacy_policy):
    scorer = make_scorer([("European", "NRP", 0.85), ("Italy", "LOCATION", 0.85)],
                         policy=legacy_policy)
    value, _signals = score(scorer, "Compare European cloud rules for banks operating in Italy.")
    assert value >= RESEARCH


def test_test_policy_turns_both_keys_on(privacy_policy):
    """tests/policy mirrors the gitops policy, which turns E and D on."""
    assert privacy_policy["ner"]["person_min_words"] == 2
    assert privacy_policy["ner"]["context_entities"] == ["NRP", "LOCATION"]


# ---------------------------------------------------------------- the false positives of the demo
@pytest.mark.parametrize("text, entities, lexicon", [
    ("Compare European and American cloud regulations for banks operating in Italy.",
     [("European", "NRP", 0.85), ("American", "NRP", 0.85), ("Italy", "LOCATION", 0.85)], 0.0),
    ("Explain why Italian banks should keep mortgage data on premises.",
     [("Italian", "NRP", 0.85)], 0.50),                                   # finance lexicon
    ("Compare Kafka and Pulsar for event streaming at scale.", [("Kafka", "PERSON", 0.85)], 0.0),
    ("Compare Raft and Paxos for a Kubernetes control plane.", [("Paxos", "PERSON", 0.85)], 0.0),
    ("Compare PostgreSQL and Cassandra for a payments platform.",
     [("Cassandra", "PERSON", 0.85)], 0.0),
    ("Explain why a bank in Milan should run its LLMs on premises.",
     [("Milan", "LOCATION", 0.85)], 0.0),
])
def test_generic_questions_go_to_sota(make_scorer, ed_policy, text, entities, lexicon):
    scorer = make_scorer(entities, policy=ed_policy)
    value, _signals = score(scorer, text)
    assert value == pytest.approx(lexicon)
    assert value < RESEARCH


def test_italian_verb_read_as_person_is_ignored(make_scorer, ed_policy):
    scorer = make_scorer([("Spiega", "PERSON", 0.85)], lang=("it", 0.95), policy=ed_policy)
    value, _signals = score(scorer, "Spiega perché un mutuo a tasso fisso conviene.")
    assert value == pytest.approx(0.50)                                   # finance only


# ---------------------------------------------------------------- real personal data still LOCAL
def test_full_name_with_nationality_and_place_stays_local(make_scorer, ed_policy):
    scorer = make_scorer([("Mario Rossi", "PERSON", 0.85), ("Italian", "NRP", 0.85),
                          ("Milan", "LOCATION", 0.85)], policy=ed_policy)
    value, signals = score(scorer, "Mario Rossi, Italian, lives in Milan.")
    assert value == pytest.approx(1 - 0.40 * 0.45 * 0.60)
    assert "(no id)" not in breakdown(scorer, signals)


def test_email_is_an_identifier_for_place(make_scorer, ed_policy):
    scorer = make_scorer([("Milan", "LOCATION", 0.85)], policy=ed_policy)
    value, signals = score(scorer, "Contact anna@example.com, based in Milan.")
    assert ("ner", "LOCATION@0.85", 0.40) in signals
    assert value == pytest.approx(1 - 0.60 * 0.60)


def test_single_word_person_is_no_identifier(make_scorer, ed_policy):
    """A dropped PERSON must not keep the nationality or the place alive."""
    scorer = make_scorer([("Kafka", "PERSON", 0.85), ("German", "NRP", 0.85),
                          ("Prague", "LOCATION", 0.85)], policy=ed_policy)
    value, _signals = score(scorer, "Kafka was a German-speaking writer in Prague.")
    assert value == 0.0


def test_structured_pii_still_decides(make_scorer, ed_policy):
    scorer = make_scorer([("Italian", "NRP", 0.85)], policy=ed_policy)
    value, _signals = score(scorer, "Italian customer, card 4111 1111 1111 1111.")
    assert value >= RESEARCH


def test_full_name_in_longer_person_entity(make_scorer, ed_policy):
    scorer = make_scorer([("Kafka", "PERSON", 0.85), ("John Smith", "PERSON", 0.70)],
                         policy=ed_policy)
    _value, signals = score(scorer, "John Smith likes Kafka.")
    assert ("ner", "PERSON@0.70", 0.60) in signals                        # best full name


# ---------------------------------------------------------------- explainability in the log
def test_ignored_entities_are_shown_with_weight_zero(make_scorer, ed_policy):
    scorer = make_scorer([("Kafka", "PERSON", 0.85), ("European", "NRP", 0.85)],
                         policy=ed_policy)
    _value, signals = score(scorer, "Compare Kafka deployments in European banks.")
    text = breakdown(scorer, signals)
    assert "PERSON@0.85(<2 words):0.00" in text
    assert "NRP@0.85(no id):0.00" in text


def test_missing_offsets_count_as_full_name(make_scorer, ed_policy):
    """Safe side: an entity without offsets is not dropped."""
    scorer = make_scorer(policy=ed_policy)

    async def presidio(_text, _lang, _timeout, _entities=None):
        return [{"entity_type": "PERSON", "score": 0.85}]

    scorer._query_presidio = presidio
    value, _signals = score(scorer, "Some text with a name in it.")
    assert value == pytest.approx(0.60)
