"""Validate generated answers for quality, specificity, and citation accuracy.

Provides:
- AnswerValidator: checks generated answers against quality thresholds
- estimate_confidence: estimates response confidence from chunks + classification
- get_response_prefix: provides appropriate hedging based on confidence
"""

import re


class AnswerValidator:
    """Validates generated answers against retrieved chunks and query context."""

    MIN_ANSWER_LENGTH = 100
    MIN_CITATIONS = 2

    MIN_RELEVANT_PHRASES = {
        "obligation": ["shall", "must", "required", "obligation", "duty", "obliged", "mandatory"],
        "definition": ["means", "refers to", "is defined as", "shall mean", "constitutes"],
        "procedural": ["step", "process", "procedure", "shall", "must", "deadline", "period"],
    }

    CELEX_PATTERN = r"3\d{4}[A-Z]\d{4}"

    STOP_WORDS = {
        "what", "is", "are", "the", "a", "an", "of", "in", "to", "for",
        "under", "by", "and", "or", "does", "do", "did", "was", "were",
        "it", "its", "this", "that", "with", "on", "at", "from", "as",
        "be", "been", "being", "have", "has", "had", "not", "no", "but",
    }

    def validate(self, query: str, answer: str, chunks: list[dict],
                 classification: dict = None) -> tuple[bool, str]:
        """Validate answer quality. Returns (passes, reason)."""
        checks = []

        # Check 1: Answer exists and is substantive
        if not answer or len(answer.strip()) < self.MIN_ANSWER_LENGTH:
            return False, "answer_too_short_or_empty"
        checks.append("has_min_length")

        # Check 2: Answer cites CELEX numbers from chunks
        chunk_celexes = {c.get("celex") for c in chunks if c.get("celex")}
        answer_celexes = set(re.findall(self.CELEX_PATTERN, answer))
        mentioned = chunk_celexes & answer_celexes
        if len(mentioned) < self.MIN_CITATIONS and len(chunk_celexes) >= self.MIN_CITATIONS:
            return False, "insufficient_citation_of_retrieved_sources"
        checks.append("has_citations")

        # Check 3: For obligation queries, verify answer contains deontic language
        if classification and classification.get("obligation_seeking"):
            has_obligation_lang = any(
                word in answer.lower()
                for word in self.MIN_RELEVANT_PHRASES["obligation"]
            )
            if not has_obligation_lang:
                return False, "obligation_query_without_obligation_language"
            checks.append("has_obligation_language")

        # Check 4: Answer addresses question keywords
        query_words = query.lower().split()
        query_keywords = {w for w in query_words if w not in self.STOP_WORDS and len(w) > 2}
        if len(query_keywords) > 2:
            answer_lower = answer.lower()
            keyword_hits = sum(1 for kw in query_keywords if kw in answer_lower)
            if keyword_hits == 0:
                return False, "answer_does_not_address_query_keywords"
            checks.append("addresses_query_keywords")

        return True, "; ".join(checks)

    def make_fallback_answer(self, query: str, chunks: list[dict],
                             classification: dict = None,
                             validation_reason: str = "") -> str:
        """Generate an informative fallback when validation fails."""
        celex_list = list(dict.fromkeys(c.get("celex") for c in chunks if c.get("celex")))
        titles = {}
        for c in chunks:
            celex = c.get("celex")
            if celex and celex not in titles:
                titles[celex] = c.get("title", "EU legislation")

        fallback_parts = [
            "I found documents related to your question, but couldn't generate a complete answer from the retrieved text.",
        ]

        if validation_reason == "obligation_query_without_obligation_language":
            fallback_parts.append(
                "The documents mention this topic but the specific employer "
                "obligation language was not found in the retrieved passages."
            )
        elif validation_reason == "insufficient_citation_of_retrieved_sources":
            fallback_parts.append(
                "The generated answer did not properly cite the specific "
                "legal provisions found in the retrieved documents."
            )

        if celex_list:
            fallback_parts.append("\nRelevant documents found:")
            for celex in celex_list[:5]:
                title = titles.get(celex, "EU legislation")
                fallback_parts.append(f"- {title} (CELEX: {celex})")

            fallback_parts.append(
                "\nTry asking a more specific question about one of these documents."
            )

        if classification and classification.get("obligation_seeking"):
            fallback_parts.append(
                "If you're looking for employer responsibilities, try including "
                "terms like 'obligations', 'duties', or 'requirements' in your question."
            )

        return "\n".join(fallback_parts)


def estimate_confidence(chunks: list[dict], classification: dict = None) -> dict:
    """Estimate confidence level for the generated answer.

    Returns a dict with:
    - level: 'high', 'medium', 'low'
    - overall_score: 0.0-1.0
    - factors: dict of contributing factors
    """
    if not chunks:
        return {"level": "low", "overall_score": 0.0, "factors": {"no_chunks": True}}

    factors = {}

    # Factor 1: Average chunk relevance score
    top_scores = [c.get("score", 0.5) for c in chunks[:5]]
    avg_score = sum(top_scores) / max(len(top_scores), 1)
    factors["relevance_score"] = max(0.0, min(1.0, 1.0 - avg_score))

    # Factor 2: Operative articles vs recitals
    article_count = sum(1 for c in chunks if c.get("article", "").startswith("art_"))
    recital_count = sum(1 for c in chunks if c.get("article", "").startswith("rct_"))
    total = article_count + recital_count
    factors["operative_ratio"] = article_count / total if total > 0 else 0.5

    # Factor 3: Deontic language presence for obligation queries
    if classification and classification.get("obligation_seeking"):
        deontic_count = 0
        deontic_words = {"shall", "must", "required", "obliged", "duty", "obligation"}
        for c in chunks[:5]:
            text_lower = c.get("text", "").lower()
            if any(w in text_lower for w in deontic_words):
                deontic_count += 1
        factors["deontic_presence"] = deontic_count / min(5, max(len(chunks), 1))

    # Calculate overall score
    weights = {"relevance_score": 0.5, "operative_ratio": 0.3, "deontic_presence": 0.2}
    present_factors = {k: v for k, v in factors.items() if k in weights}
    if present_factors:
        total_weight = sum(weights[k] for k in present_factors)
        overall = sum(factors[k] * weights[k] for k in present_factors) / total_weight
    else:
        overall = 0.5

    overall = max(0.0, min(1.0, overall))

    if overall >= 0.7:
        level = "high"
    elif overall >= 0.4:
        level = "medium"
    else:
        level = "low"

    return {"level": level, "overall_score": round(overall, 3), "factors": factors}


def get_response_prefix(confidence: dict) -> str:
    """Get appropriate hedging prefix based on confidence level."""
    if confidence["level"] == "high":
        return "Based on the retrieved EU law documents, "
    elif confidence["level"] == "medium":
        return "Based on the available legal texts, "
    else:
        return "Based on partial information from related documents, "
