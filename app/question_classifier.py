"""Legal question classifier for EU law queries.

Classifies queries by:
- Question type (wh_question, polar_question, statement, command, request)
- Legal information intent (obligation, definition, procedural, temporal, entity)
- Legal actors (employers, member states, etc.)
- Whether clarification is needed
"""

import re


class EUQuestionClassifier:
    """Classifies EU law queries for better retrieval and response strategy."""

    # Question type patterns
    WH_PATTERNS = [
        r"^(what|why|when|where|which|who|whom|whose|how)\b",
        r"^(what are|what is|what's|what were|what was)\b",
        r"^(how does|how do|how is|how are|how can|how to)\b",
        r"^(why (is|are|do|does|did|would|should))\b",
        r"^(when (is|are|do|does|did|will|would|should))\b",
        r"^(where (is|are|do|does|did|can))\b",
        r"^(who (is|are|was|were|does|do))\b",
    ]

    POLAR_PATTERNS = [
        r"^(is|are|was|were|do|does|did|has|have|had|can|could|will|would|shall|should|may|might)\b",
        r"^(does|do|did)\s+\w+\s+(apply|cover|include|require|prohibit|allow|permit)\b",
        r"^(is|are)\s+(there|it)\b",
    ]

    COMMAND_PATTERNS = [
        r"^(list|show|tell|give|find|search|explain|describe|summarize|outline|detail)\b",
        r"^(compare|contrast|analyze|evaluate|discuss)\b",
    ]

    REQUEST_PATTERNS = [
        r"^(can you|could you|would you|will you|please)\b",
        r"^i (want|need|would like|am looking for)\b",
        r"^i'd like\b",
    ]

    # EU law-specific patterns for obligation/responsibility detection
    OBLIGATION_KEYWORDS = {
        "responsibilities", "obligations", "duties", "requirements",
        "must", "shall", "required", "mandatory", "comply",
        "reporting", "disclosure", "transparency",
        "obligation", "responsibility", "duty", "requirement",
        "compliant", "compliance", "non-compliance",
        "liable", "liability", "penalty", "sanction",
        "prohibited", "forbidden", "not permitted",
        "allowed", "permitted", "authorized",
        "obliged", "bound to", "mandated",
        "ensuring", "shall ensure", "must ensure",
    }

    DEFINITION_KEYWORDS = {
        "definition", "meaning", "concept", "what is", "what are",
        "define", "constitutes", "scope", "applies to",
        "covered by", "falls under", "within the scope",
    }

    ACTOR_KEYWORDS = {
        "employer": ["employer", "employers", "organisation", "organisations", "undertaking", "undertakings"],
        "company": ["company", "companies", "corporation", "corporations", "enterprise", "enterprises", "firm", "firms", "business", "businesses"],
        "employee": ["employee", "employees", "worker", "workers", "staff", "personnel", "workforce"],
        "member_state": ["member state", "member states", "national authority", "national authorities", "member-state"],
        "commission": ["european commission", "commission", "eu commission"],
        "controller": ["controller", "controllers", "data controller", "data controllers"],
        "processor": ["processor", "processors", "data processor", "data processors"],
        "supervisory_authority": ["supervisory authority", "supervisory authorities", "data protection authority"],
        "public_authority": ["public authority", "public authorities", "government", "government body", "regulator", "regulatory body"],
        "citizen": ["citizen", "citizens", "individual", "individuals", "person", "persons", "natural person", "data subject"],
    }

    def __init__(self):
        self._classifier = None

    def _detect_sentence_type(self, query: str) -> dict:
        """Detect whether the query is a question, command, request, or statement."""
        query_stripped = query.strip()

        # Check question patterns
        for pattern in self.WH_PATTERNS:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                return {"is_question": True, "type": "wh_question"}

        for pattern in self.POLAR_PATTERNS:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                return {"is_question": True, "type": "polar_question"}

        if query_stripped.endswith("?"):
            return {"is_question": True, "type": "polar_question"}

        # Check command patterns
        for pattern in self.COMMAND_PATTERNS:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                return {"is_question": False, "type": "command"}

        # Check request patterns
        for pattern in self.REQUEST_PATTERNS:
            if re.search(pattern, query_stripped, re.IGNORECASE):
                return {"is_question": False, "type": "request"}

        # Default: statement
        return {"is_question": False, "type": "statement"}

    def _detect_legal_intent(self, query: str) -> str:
        """Detect the legal information intent of the query."""
        query_lower = query.lower()

        # Check for obligation/responsibility patterns
        obligation_patterns = [
            r"(what are|what's|describe|explain|list|identify).+(responsib|obliga|dut|requirement|must|shall)",
            r"(responsib|obliga|dut).+(under|pursuant|according|following|of)",
            r"(how|what).+(comply|report|disclos|transparen)",
            r"(must|shall|required|obliged|mandated)\s+\w+\s+(under|by|pursuant|according)",
            r"(obligation|duty|responsibility|requirement)\s+(of|for|under|imposed)",
            r"(what|which).+(obligation|duty|responsibility|requirement)",
            r"(obligations|duties|responsibilities|requirements)\s+(of|for|of the|for the)",
            r"(employer|company|undertaking|organisation)\w*\s+(must|shall|required|obliged|has to|have to)",
            r"(prohibited|forbidden|not allowed|not permitted)\s+(under|by|pursuant)",
            r"(allowed|permitted|authorized)\s+(under|by|pursuant)",
        ]
        for pattern in obligation_patterns:
            if re.search(pattern, query_lower):
                return "obligation"

        # Check for keyword counts
        obligation_word_count = sum(
            1 for kw in self.OBLIGATION_KEYWORDS if kw in query_lower
        )
        if obligation_word_count >= 2:
            return "obligation"

        # Definition patterns
        definition_patterns = [
            r"(what is|what's|define|definition|meaning|concept).+(under|in|according to|pursuant)",
            r"(what is|what's|define|definition|meaning).+(directive|regulation|act|law|article)",
            r"(define|explain|describe)\s+(the\s+)?(concept|term|definition|meaning)\s+(of\s+)?",
            r"(what|which)\s+(is|are)\s+(the\s+)?(definition|scope|purpose|objective)\s+(of\s+)?",
            r"(scope|coverage|application)\s+(of\s+)?(the\s+)?(directive|regulation)",
        ]
        for pattern in definition_patterns:
            if re.search(pattern, query_lower):
                return "definition"

        definition_word_count = sum(
            1 for kw in self.DEFINITION_KEYWORDS if kw in query_lower
        )
        if definition_word_count >= 2:
            return "definition"

        # Procedural patterns
        procedural_patterns = [
            r"(how|what steps|what process|what procedure|what requirements).+(to|for)",
            r"(what|which).+(procedur|process|step|method)",
            r"(how to|how do i|how does one|how can i)",
            r"(steps|procedure|process|method)\s+(for|to|of|required)",
            r"(deadline|time[-\s]?limit|period|timeline|when to|when must)",
            r"(effective date|entry into force|application date|transposition)",
        ]
        for pattern in procedural_patterns:
            if re.search(pattern, query_lower):
                return "procedural"

        # Temporal patterns
        temporal_patterns = [
            r"(when|what date|what deadline|effective date|comes into force|enters into)",
            r"(timeline|timeframe|schedule|period|duration)",
            r"(until|before|after|as of|from)\s+\d{4}",
            r"(currently|now|present|future|upcoming|forthcoming)",
        ]
        for pattern in temporal_patterns:
            if re.search(pattern, query_lower):
                return "temporal"

        # Entity queries
        entity_patterns = [
            r"(what|which)\s+(directive|regulation|act|law|article|clause|section)",
            r"(tell|show|list|find)\s+(me\s+)?(about|the|all|some)",
            r"(what|which)\s+(agenc|body|authority|institution|organ)",
            r"(celex|document|reference|number)\s+\d",
        ]
        for pattern in entity_patterns:
            if re.search(pattern, query_lower):
                return "entity"

        return "entity"

    def _extract_legal_actors(self, query: str) -> list[str]:
        """Extract legal actors (parties with obligations/rights) from the query."""
        found = []
        query_lower = query.lower()

        for actor_type, keywords in self.ACTOR_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if actor_type not in found:
                        found.append(actor_type)
                    break

        return found

    def _estimate_confidence(self, query: str, result: dict) -> float:
        """Estimate confidence in the classification (0.0 to 1.0)."""
        score = 0.5  # Base confidence

        # Higher confidence if it's clearly a question
        if result.get("is_question"):
            score += 0.15

        # Higher confidence if legal intent is clear
        if result.get("legal_intent") != "entity":
            score += 0.15

        # Higher confidence if legal actors found
        if result.get("legal_actors"):
            score += 0.1

        # Higher confidence if EU law keywords present
        eu_keywords = {"gdpr", "directive", "regulation", "eu law", "european union",
                       "pay transparency", "ai act", "data protection", "general data protection",
                       "celex", "article", "chapter", "annex", "recital"}
        query_lower = query.lower()
        if any(kw in query_lower for kw in eu_keywords):
            score += 0.1

        return min(score, 1.0)

    def classify(self, query: str) -> dict:
        """Classify query into question type, legal intent, and extracted actors.

        Args:
            query: The user's query string.

        Returns:
            Dictionary with classification results.
        """
        query_stripped = query.strip()

        result = {
            "raw_query": query_stripped,
            "is_question": False,
            "question_type": None,
            "legal_intent": None,
            "legal_actors": [],
            "obligation_seeking": False,
            "confidence": 0.0,
        }

        # Sentence type detection
        sentence_info = self._detect_sentence_type(query_stripped)
        result["is_question"] = sentence_info["is_question"]
        result["question_type"] = sentence_info["type"]

        # Legal intent detection
        result["legal_intent"] = self._detect_legal_intent(query_stripped)
        result["obligation_seeking"] = result["legal_intent"] == "obligation"

        # Legal actor extraction
        result["legal_actors"] = self._extract_legal_actors(query_stripped)

        # Confidence estimation
        result["confidence"] = self._estimate_confidence(query_stripped, result)

        return result

    def needs_clarification(self, classification: dict) -> bool:
        """Determine if the system should ask for clarification."""
        # If query is not a question and no clear legal intent
        if not classification.get("is_question") and classification.get("legal_intent") == "entity":
            # Check if it contains EU law keywords that suggest implicit question
            eu_keywords = {"gdpr", "ai act", "directive", "regulation",
                           "eu law", "european union", "pay transparency"}
            query_lower = classification.get("raw_query", "").lower()
            if not any(kw in query_lower for kw in eu_keywords):
                return True
        return False

    def should_answer(self, classification: dict, search_results: list[dict],
                      min_confidence: float = 0.3) -> tuple:
        """Determine if we have sufficient confidence to answer.

        Args:
            classification: Result from classify()
            search_results: List of search result chunks
            min_confidence: Minimum confidence threshold

        Returns:
            (should_answer: bool, reason: str)
        """
        # Must have search results
        if not search_results:
            return False, "no_relevant_documents"

        # Must meet minimum confidence
        if classification.get("confidence", 0) < min_confidence:
            return False, "low_classification_confidence"

        # Must have reasonable relevance scores
        top_scores = [c.get("score", 0) for c in search_results[:3]]
        avg_score = sum(top_scores) / max(len(top_scores), 1)
        if avg_score < 0.45:
            return False, "low_relevance_scores"

        # For obligation questions, must find deontic language in results
        if classification.get("obligation_seeking"):
            deontic_keywords = {"shall", "must", "required", "obliged", "duty",
                                "obligation", "responsibility", "prohibited",
                                "permitted", "mandatory", "comply"}
            has_obligation_language = False
            for chunk in search_results[:5]:
                chunk_text_lower = chunk.get("text", "").lower()
                if any(kw in chunk_text_lower for kw in deontic_keywords):
                    has_obligation_language = True
                    break
            if not has_obligation_language:
                return False, "insufficient_obligation_language"

        return True, "confident"
