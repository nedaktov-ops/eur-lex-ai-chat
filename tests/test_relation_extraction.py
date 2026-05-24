"""Tests for legal relation extraction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from relation_extractor import extract_legal_relations, summarize_relations


def test_obligation_extraction():
    text = "Employers shall disclose salary ranges to job applicants. The data controller must protect personal data."
    relations = extract_legal_relations(text)
    assert len(relations["obligations"]) >= 1


def test_prohibition_extraction():
    text = "Employers shall not ask about previous salary history. The controller shall not process data without consent."
    relations = extract_legal_relations(text)
    assert len(relations["prohibitions"]) >= 1


def test_right_extraction():
    text = "Workers have the right to request pay information. Data subjects may request access to their data."
    relations = extract_legal_relations(text)
    assert len(relations["rights"]) >= 1


def test_actor_extraction():
    text = "Employers must disclose pay to workers. The data controller shall inform the data subject."
    relations = extract_legal_relations(text)
    assert any("employer" in a for a in relations["actors"])


def test_summarize_relations():
    text = "Employers shall disclose salary ranges. Workers have the right to request pay information."
    relations = extract_legal_relations(text)
    summary = summarize_relations(relations)
    assert "OBLIGATIONS" in summary or "RIGHTS" in summary or "ACTORS" in summary
