"""Tests for discourse-aware scoring."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from search import discourse_boost


def test_operative_article_boost_over_recital():
    article_chunk = {"celex": "32023L0970", "article": "art_5", "text": "Employers shall disclose salary ranges. They must ensure pay transparency."}
    recital_chunk = {"celex": "32023L0970", "article": "rct_10", "text": "The principle of equal pay is fundamental to the European social model."}
    ctx = {"obligation_seeking": True}
    article_score = discourse_boost(article_chunk, ctx)
    recital_score = discourse_boost(recital_chunk, ctx)
    assert article_score > recital_score


def test_deontic_boost_for_obligation_queries():
    chunk = {"celex": "32023L0970", "article": "art_5", "text": "Employers shall disclose salary ranges."}
    ctx = {"obligation_seeking": True}
    score = discourse_boost(chunk, ctx)
    assert score > 1.0


def test_no_boost_for_non_obligation():
    chunk = {"celex": "32023L0970", "article": "art_5", "text": "The definition of pay shall mean ordinary wage."}
    ctx = {"obligation_seeking": False}
    score = discourse_boost(chunk, ctx)
    # Operative article boost (1.3) applies regardless of query context
    assert score == 1.3


def test_none_article_code_does_not_crash():
    chunk = {"celex": "32023L0970", "article": None, "text": "Some legal text here."}
    ctx = {"obligation_seeking": False}
    score = discourse_boost(chunk, ctx)
    assert score == 1.0


def test_missing_article_code_does_not_crash():
    chunk = {"celex": "32023L0970", "text": "Some legal text here."}
    ctx = {"obligation_seeking": False}
    score = discourse_boost(chunk, ctx)
    assert score == 1.0
