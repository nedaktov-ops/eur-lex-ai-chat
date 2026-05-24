"""Tests for rag.build_prompt and answer_question."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag import build_prompt, ENSURE_CITATION_PROMPT, SYSTEM_PROMPT

def test_build_prompt_no_duplicate_system():
    """User prompt should not start with 'System:'."""
    chunks = [
        {"celex": "32023L0970", "title": "Pay Transparency Directive", "article": "art_1", "text": "Employers shall provide salary ranges."},
    ]
    prompt = build_prompt("What are obligations?", chunks, classification=None, extra_system_notes=None)
    # Should not contain "System:" at the beginning
    assert not prompt.startswith("System:"), "User prompt should not duplicate SYSTEM_PROMPT"
    # Should contain context and query
    assert "Relevant excerpts from EU law documents:" in prompt
    assert "What are obligations?" in prompt

def test_build_prompt_with_extra_notes():
    """When extra_system_notes provided, they appear at the top."""
    chunks = [
        {"celex": "32023L0970", "title": "Pay Transparency Directive", "article": "art_1", "text": "Employers shall provide salary ranges."},
    ]
    extra = "Please be concise."
    prompt = build_prompt("What are obligations?", chunks, classification=None, extra_system_notes=extra)
    assert prompt.startswith("Please be concise."), "Extra notes should appear first"

def test_ensure_citation_prompt_only_on_retry(monkeypatch):
    """answer_question should only add ENSURE_CITATION_PROMPT on retry."""
    from app import rag as rag_module
    called_with = {}
    def mock_build_prompt(query, context_chunks, classification=None, extra_system_notes=None):
        called_with['extra'] = extra_system_notes
        return "mock prompt"
    monkeypatch.setattr(rag_module, 'build_prompt', mock_build_prompt)
    monkeypatch.setattr(rag_module, 'call_groq', lambda prompt: {"answer": "test", "prompt_tokens": 10, "completion_tokens": 5, "duration_ms": 100})
    # First attempt
    rag_module.answer_question("q", [], retry_with_citation_emphasis=False)
    assert called_with.get('extra') is None, "First attempt should not include ENSURE_CITATION_PROMPT"
    # Retry attempt
    rag_module.answer_question("q", [], retry_with_citation_emphasis=True)
    assert called_with.get('extra') is not None
    assert "CRITICAL" in called_with['extra']
