"""Tests for query expansion with legal synonyms."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from query_expander import expand_query, expand_obligation_query, AutoExpander


def test_obligation_query_expansion():
    expansions = expand_obligation_query("what are the responsibilities of employers under pay transparency")
    assert len(expansions) >= 3
    assert any("obligations" in e.lower() for e in expansions)
    assert any("undertaking" in e.lower() for e in expansions)


def test_non_obligation_queries_not_overexpanded():
    expansions = expand_query("what is GDPR")
    assert len(expansions) <= 5


def test_original_query_preserved():
    q = "employer responsibilities under the AI Act"
    expansions = expand_query(q)
    assert expansions[0] == q


def test_pay_synonyms_applied():
    expansions = expand_obligation_query("salary disclosure obligations for employers")
    any_pay = any("pay" in e.lower() or "remuneration" in e.lower() for e in expansions)
    assert any_pay


def test_auto_expander_records_failure():
    exp = AutoExpander()
    exp.record_failure("what are employer obligations under GDPR", "insufficient_citation")
    auto = exp.get_auto_expansions()
    assert "obligations" in auto
