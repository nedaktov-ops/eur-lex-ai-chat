"""Tests for extract_citations in rag.py."""
import sys
import os
# Add project root to path to import app package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.rag import extract_citations

def test_extract_with_celex_prefix():
    text = "According to CELEX 32023L0970 and CELEX 32018L1972, employers must..."
    assert extract_citations(text) == ["32023L0970", "32018L1972"]

def test_extract_without_prefix():
    text = "Directive 32023L0970 requires pay transparency. Also see 32018L1972."
    assert extract_citations(text) == ["32023L0970", "32018L1972"]

def test_extract_mixed():
    # Even without a space after CELEX, the number substring should be captured.
    text = "Under CELEX32023L0970 and 32018L1972, obligations apply."
    assert extract_citations(text) == ["32023L0970", "32018L1972"]

def test_extract_none():
    text = "No citations here."
    assert extract_citations(text) == []

def test_extract_obfuscated():
    # Ensure we don't match random numbers like 12345
    text = "The year 2023 is not a CELEX. But 32023L0970 is."
    assert extract_citations(text) == ["32023L0970"]
