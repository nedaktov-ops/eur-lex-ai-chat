"""Shared fixtures for build_index tests."""

import json
import os
import pytest


SAMPLE_XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:eli="http://data.europa.eu/eli/ontology"
      xmlns:oj="http://publications.europa.eu/resource/authority/oj">
<head><title>Test Directive</title></head>
<body>
<div class="eli-main-title">
    <span class="oj-doc-ti">Directive (EU) 2024/1234 on Test Matters</span>
</div>
<div class="eli-container">
    <div class="eli-subdivision" id="pbl_1">
        <p>THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,</p>
        <p>Having regard to the Treaty on the Functioning of the European Union,</p>
    </div>
    <div class="eli-subdivision" id="art_1">
        <p>Article 1</p>
        <p>Subject matter</p>
        <p>This Directive establishes rules for testing purposes.</p>
    </div>
    <div class="eli-subdivision" id="art_2">
        <p>Article 2</p>
        <p>Definitions</p>
        <p>For the purposes of this Directive, 'test' means any evaluation procedure.</p>
    </div>
    <div class="eli-subdivision" id="ann_1">
        <p>ANNEX I</p>
        <p>Test methods referred to in Article 1 of this Directive concerning the evaluation of conformity assessment procedures and certification requirements.</p>
    </div>
</div>
</body>
</html>"""


SAMPLE_CORRIGENDUM_XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Corrigendum to Test Directive</title></head>
<body>
<div id="documentView">
    <p>On page 5, in Article 1(2), the date '2025' is replaced by '2026'.</p>
    <p>This corrigendum concerns all language versions.</p>
</div>
</body>
</html>"""


SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"


@pytest.fixture
def mock_sparql_response():
    """Mock SPARQL JSON response with a mix of normal docs and corrigenda."""
    return {
        "results": {
            "bindings": [
                {
                    "doc": {"type": "uri", "value": "http://data.europa.eu/eli/dir/2024/1234"},
                    "type": {"type": "uri", "value": "http://publications.europa.eu/resource/authority/resource-type/DIR"},
                    "celex": {"type": "literal", "value": "32024L1234"},
                    "date": {"type": "literal", "value": "2024-06-15"},
                },
                {
                    "doc": {"type": "uri", "value": "http://data.europa.eu/eli/reg/2023/567"},
                    "type": {"type": "uri", "value": "http://publications.europa.eu/resource/authority/resource-type/REG"},
                    "celex": {"type": "literal", "value": "32023R0567"},
                    "date": {"type": "literal", "value": "2023-01-20"},
                },
                # This corrigendum should be filtered out by SPARQL FILTER
                {
                    "doc": {"type": "uri", "value": "http://data.europa.eu/eli/dir/2024/1234/corrigendum/1"},
                    "type": {"type": "uri", "value": "http://publications.europa.eu/resource/authority/resource-type/DIR"},
                    "celex": {"type": "literal", "value": "32024L1234R(01)"},
                    "date": {"type": "literal", "value": "2024-08-01"},
                },
            ]
        }
    }


@pytest.fixture
def chunks_for_test():
    """Small set of chunks for testing embedding/FAISS/db pipeline."""
    return [
        {"text": "This Directive establishes rules for testing purposes.",
         "celex": "32024L1234", "title": "Directive on Test Matters",
         "article": "art_1", "type": "article"},
        {"text": "For the purposes of this Directive, 'test' means any evaluation procedure.",
         "celex": "32024L1234", "title": "Directive on Test Matters",
         "article": "art_2", "type": "article"},
        {"text": "This Regulation sets out harmonised rules for artificial intelligence.",
         "celex": "32023R0567", "title": "AI Act Regulation",
         "article": "art_2", "type": "article"},
    ]
