"""Tests for answer validation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from answer_validator import AnswerValidator, estimate_confidence


def test_validates_short_answers():
    validator = AnswerValidator()
    passes, reason = validator.validate(
        query="What is GDPR?",
        answer="It's a law.",
        chunks=[{"celex": "32016R0679"}],
        classification={"obligation_seeking": False},
    )
    assert not passes
    assert reason == "answer_too_short_or_empty"


def test_passes_good_answers():
    validator = AnswerValidator()
    passes, reason = validator.validate(
        query="What is GDPR?",
        answer="The General Data Protection Regulation (GDPR, Regulation EU 2016/679, CELEX 32016R0679) is an EU law that protects personal data processing.",
        chunks=[{"celex": "32016R0679"}],
        classification={"obligation_seeking": False},
    )
    assert passes


def test_obligation_without_deontic_language_fails():
    validator = AnswerValidator()
    passes, reason = validator.validate(
        query="What are employer obligations?",
        answer="The directive mentions some things about pay. CELEX 32023L0970 covers this topic. CELEX 32019R0817 also applies.",
        chunks=[{"celex": "32023L0970"}, {"celex": "32019R0817"}],
        classification={"obligation_seeking": True},
    )
    assert not passes
    assert "obligation" in reason


def test_insufficient_citation_fails():
    validator = AnswerValidator()
    passes, reason = validator.validate(
        query="What are employer obligations?",
        answer="Employers shall comply with all requirements under the Pay Transparency Directive. CELEX 32023L0970 covers this topic. Employers must ensure compliance with the directive's provisions regarding pay transparency and reporting obligations. Companies shall disclose information to the relevant authorities.",
        chunks=[{"celex": "32023L0970"}, {"celex": "32019R0817"}, {"celex": "32016R0679"}],
        classification={"obligation_seeking": True},
    )
    assert not passes
    assert "insufficient_citation" in reason


def test_make_fallback_answer():
    validator = AnswerValidator()
    fallback = validator.make_fallback_answer(
        query="What are employer obligations?",
        chunks=[{"celex": "32023L0970", "title": "Pay Transparency Directive"}],
        classification={"obligation_seeking": True},
        validation_reason="insufficient_citation_of_retrieved_sources",
    )
    assert "32023L0970" in fallback
    assert "obligation" in fallback.lower() or "responsibilities" in fallback.lower()


def test_estimate_confidence():
    chunks = [
        {"celex": "32023L0970", "article": "art_5", "text": "Employers shall disclose salary ranges. They must ensure pay transparency.", "score": 0.6},
        {"celex": "32023L0970", "article": "art_12", "text": "Data protection for workers pay data shall be ensured.", "score": 0.5},
    ]
    conf = estimate_confidence(chunks, classification={"obligation_seeking": True})
    assert "level" in conf
    assert "overall_score" in conf
    assert conf["level"] in ("high", "medium", "low")
