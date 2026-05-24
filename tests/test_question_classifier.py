"""Tests for the EU law question classifier."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from question_classifier import EUQuestionClassifier


def test_obligation_intent_detected():
    clf = EUQuestionClassifier()
    result = clf.classify("What are the responsibilities of employers under the Pay Transparency Directive?")
    assert result["legal_intent"] == "obligation"
    assert result["obligation_seeking"] is True


def test_definition_intent_detected():
    clf = EUQuestionClassifier()
    result = clf.classify("What is GDPR?")
    assert result["legal_intent"] in ("definition", "entity")


def test_actor_extraction():
    clf = EUQuestionClassifier()
    result = clf.classify("What are employer obligations under the AI Act?")
    assert "employer" in result.get("legal_actors", [])


def test_non_question_classified():
    clf = EUQuestionClassifier()
    result = clf.classify("GDPR")
    assert result["is_question"] is False


def test_plain_statement_does_not_need_clarification():
    clf = EUQuestionClassifier()
    result = clf.classify("GDPR")
    assert clf.needs_clarification(result) is False
