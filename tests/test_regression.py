"""Regression tests — verify previously working queries still pass."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from question_classifier import EUQuestionClassifier
from query_expander import expand_obligation_query, expand_query
from search import discourse_boost
from relation_extractor import extract_legal_relations
from answer_validator import AnswerValidator


def test_question_classifier_pay_transparency():
    clf = EUQuestionClassifier()
    r = clf.classify("What are the responsibilities of employers under the Pay Transparency Directive?")
    assert r["legal_intent"] == "obligation"
    assert "employer" in r.get("legal_actors", [])


def test_query_expansion_includes_obligation_variants():
    expansions = expand_obligation_query("what are the responsibilities of employers under pay transparency")
    assert len(expansions) >= 3
    all_text = " ".join(e.lower() for e in expansions)
    assert "obligation" in all_text or "duties" in all_text


def test_discourse_boost_operative_over_recital():
    art = {"celex": "32023L0970", "article": "art_5", "text": "Employers shall disclose salary ranges."}
    rct = {"celex": "32023L0970", "article": "rct_10", "text": "The principle of equal pay is fundamental."}
    ctx = {"obligation_seeking": True}
    assert discourse_boost(art, ctx) > discourse_boost(rct, ctx)


def test_relation_extraction_finds_obligations():
    text = "Employers shall disclose salary ranges. Employers must ensure pay transparency."
    r = extract_legal_relations(text)
    assert len(r["obligations"]) >= 1


def test_answer_validation_rejects_empty():
    v = AnswerValidator()
    passes, reason = v.validate("test", "", [], {})
    assert not passes
    assert "answer_too_short" in reason


def test_confidence_estimation_returns_valid():
    from answer_validator import estimate_confidence
    chunks = [{"celex": "32023L0970", "article": "art_5", "text": "Employers shall comply.", "score": 0.6}]
    c = estimate_confidence(chunks, {"obligation_seeking": True})
    assert "level" in c
    assert "overall_score" in c
