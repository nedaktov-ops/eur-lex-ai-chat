# Phase 4: Continuous Improvement Loop — Implementation Plan

> **For agentic workers:** Execute tasks in this session inline.

**Goal:** Establish automated retry on validation failure, log analysis for insight generation, and auto-expansion of query dictionaries based on failure patterns.

**Architecture:** Three independent subsystems: (1) LLM retry in main.py when answer validation fails, (2) feedback_analyzer.py to mine pipeline logs for patterns, (3) auto-expansion from failures into query_expander.py. All leverage existing Phase 0 pipeline logging.

**Tech Stack:** Python 3.12, FastAPI, json (log analysis), existing pipeline logs

---

### Task 1: LLM Retry on Validation Failure

**Files:**
- Modify: `backend/main.py:254-284`
- Modify: `backend/rag.py:89-95`

**Problem:** The LLM (Llama 3.3 70B on Groq) is non-deterministic — sometimes generates detailed answers with CELEX citations, sometimes short vague ones. When validation fails, instead of immediately returning a fallback, retry once with an emphasized instruction to include CELEX numbers.

- [ ] **Step 1: Add ENSURE_CITATION_PROMPT to rag.py**

Add a secondary system prompt snippet that strongly emphasizes CELEX citation.

In `backend/rag.py`, add a constant after `build_prompt()`:

```python
ENSURE_CITATION_PROMPT = """
CRITICAL: Your answer MUST include the CELEX number (e.g., 32023L0970) for each source you cite.
Every claim must be attributed to a specific CELEX document.
If you use information from a provided context chunk, cite its CELEX number inline.
"""
```

- [ ] **Step 2: Modify main.py retry logic**

In `backend/main.py`, replace the current validation failure block to retry once:

```python
    if not passes_validation:
        logger.warning(f"Answer validation failed: {validation_reason} | query: {query[:80]}")
        # Retry once with emphasis on CELEX citation
        logger.info(f"Retrying with CELEX citation emphasis for request {request_id}")
        result2 = answer_question(
            query, chunks, request_id=request_id,
            classification=classification, retry_with_citation_emphasis=True,
        )
        passes_validation2, validation_reason2 = validator.validate(
            query=query, answer=result2.get("answer", ""),
            chunks=chunks, classification=classification,
        )
        if passes_validation2:
            result = result2
            passes_validation = True
            validation_reason = validation_reason2
            logger.info(f"Retry succeeded for request {request_id}")
```

But if the retry also fails, fall back to the original fallback logic.

- [ ] **Step 3: Modify rag.py answer_question signature**

Update `answer_question()` to accept an optional `retry_with_citation_emphasis` parameter. When True, append `ENSURE_CITATION_PROMPT` to the system instructions.

- [ ] **Step 4: Test the retry**

Run the Pay Transparency obligation query. Without retry it fails ~50% of the time. With retry, success rate should approach ~75%.

---

### Task 2: Feedback Analyzer

**Files:**
- Create: `scripts/feedback_analyzer.py`

- [ ] **Step 1: Write feedback_analyzer.py**

Script that reads pipeline logs (from server stdout, or from a log file), extracts patterns:
- Queries that got fallback responses (with reason)
- Queries with low confidence
- Most common validation failure reasons
- Recent failed queries

Outputs: JSON report with insights.

```python
#!/usr/bin/env python3
"""Analyze pipeline logs for improvement opportunities."""
import json, sys, os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

def analyze_logs(log_sources=None, days=7):
    """Analyze pipeline logs from multiple sources."""
    if log_sources is None:
        log_sources = sys.stdin
    
    stats = {
        "total_queries": 0,
        "fallback_queries": [],
        "low_confidence_queries": [],
        "validation_failures": Counter(),
        "pipeline_breakdown": defaultdict(int),
    }
    
    for line in log_sources:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        stage = entry.get("stage")
        if stage:
            stats["pipeline_breakdown"][stage] += 1
        
        data = entry.get("data", {})
        
        if stage == "answer_generated":
            stats["total_queries"] += 1
            if data.get("validation_passed") is False:
                stats["validation_failures"][data.get("citations_count", 0)] += 1
                stats["fallback_queries"].append(data)
            elif data.get("confidence_level") in ("low", "medium"):
                stats["low_confidence_queries"].append(data)
    
    return stats
```

- [ ] **Step 2: Test**

Run: `python3 scripts/feedback_analyzer.py < /tmp/server_phase3g.log`

Should output a JSON report with query counts and patterns.

---

### Task 3: Auto-Expansion from Failed Queries

**Files:**
- Modify: `backend/query_expander.py`

- [ ] **Step 1: Add auto-expansion class**

Add `AutoExpander` class that records failed-query terms and can suggest new synonym pairs.

- [ ] **Step 2: Integrate with main.py**

When validation fails, extract key terms from the query and record them for potential expansion.

---

### Task 4: Phase 4 Checkpoint

- [ ] **Step 1: Save checkpoint**

Run: `python3 scripts/checkpoint_save.py --phase 4 --message "Phase 4: retry strategy + feedback analyzer + auto-expansion"`
