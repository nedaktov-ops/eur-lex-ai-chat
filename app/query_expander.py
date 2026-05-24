"""Query expansion for EU law plain-language queries.

Maps common expressions to legal terminology found in EUR-LEX documents.
This is essential because the FAISS index is built from legal text, not conversational English.
"""

import re
from typing import Dict, List, Optional

# Core term mappings (built from Legal-BERT vocabulary analysis)
LEGAL_SYNONYMS: Dict[str, List[str]] = {
    # Employer-related
    "employer": ["undertaking", "company", "organization", "legal person", "economic operator"],
    "company": ["undertaking", "organization", "enterprise", "legal person", "economic operator"],
    "business": ["undertaking", "enterprise", "economic operator", "commercial entity"],
    "organisation": ["organization", "undertaking", "body", "entity", "institution"],

    # Responsibility-related
    "responsibilities": ["obligations", "duties", "requirements", "compliance obligations", "responsibilities"],
    "obligations": ["duties", "requirements", "obligations imposed on", "obligations under"],
    "duties": ["obligations", "requirements", "responsibilities", "tasks"],
    "requirements": ["conditions", "obligations", "prerequisites", "criteria", "stipulations"],

    # Pay transparency specific
    "salary": ["pay", "remuneration", "compensation", "wage", "earnings", "income"],
    "pay": ["remuneration", "salary", "compensation", "earnings", "wage"],
    "wage": ["pay", "remuneration", "salary", "compensation", "earnings"],
    "disclosure": ["transparency", "reporting", "publication", "communication", "notification"],
    "salary disclosure": ["pay transparency", "remuneration reporting", "pay disclosure", "compensation transparency"],
    "pay gap": ["gender pay gap", "pay differential", "remuneration gap", "wage gap"],
    "equal pay": ["equal remuneration", "pay equality", "equal treatment in pay", "non-discrimination in pay"],

    # Regulatory action
    "comply with": ["meet the requirements of", "fulfill obligations under", "adhere to", "satisfy"],
    "regulated by": ["governed by", "subject to", "within the scope of", "falling under"],
    "allowed": ["permitted", "authorized", "not prohibited", "admissible", "lawful"],
    "forbidden": ["prohibited", "not permitted", "restricted", "banned", "disallowed"],
    "apply to": ["cover", "extend to", "be applicable to", "concern", "relate to"],

    # Time references
    "when does": ["date of application", "entry into force", "effective date", "transposition deadline"],
    "deadline": ["time limit", "period", "transposition date", "implementation deadline", "compliance date"],
    "effective date": ["date of application", "entry into force", "implementation date"],
    "come into force": ["enter into force", "become effective", "take effect", "apply from"],

    # General legal
    "law": ["legislation", "regulation", "directive", "act", "legal instrument", "norm"],
    "legal": ["lawful", "legitimate", "statutory", "regulatory", "legislative"],
    "rule": ["regulation", "provision", "stipulation", "requirement", "norm"],
    "right": ["entitlement", "claim", "entitlement under", "right conferred by"],
    "protect": ["safeguard", "ensure", "guarantee", "preserve", "shield"],
    "violation": ["infringement", "breach", "non-compliance", "contravention", "offence"],
    "penalty": ["sanction", "fine", "penalty", "consequence", "remedy"],
    "exemption": ["derogation", "exception", "exclusion", "carve-out", "waiver"],
}


def expand_query(query: str) -> List[str]:
    """Expand a plain-language query with legal synonyms.

    Returns a list of query variations to improve search recall.
    The original query is always first.
    """
    query_original = query.strip()
    query_lower = query_original.lower()
    variations = [query_original]

    # Check for multi-word phrases first (longer phrases have priority)
    words = query_lower.split()
    for phrase_len in range(3, 1, -1):
        for i in range(len(words) - phrase_len + 1):
            phrase = " ".join(words[i:i+phrase_len])
            if phrase in LEGAL_SYNONYMS:
                synonyms = LEGAL_SYNONYMS[phrase]
                for syn in synonyms:
                    if syn != phrase:
                        # Replace the phrase in the original query (case-insensitive)
                        new_query = _replace_insensitive(query_original, phrase, syn)
                        if new_query != query_original:
                            variations.append(new_query)

    # Single-word replacements
    for word in set(words):
        if word in LEGAL_SYNONYMS:
            synonyms = LEGAL_SYNONYMS[word]
            for syn in synonyms:
                if syn != word and syn not in query_lower:
                    new_query = _replace_insensitive(query_original, word, syn)
                    if new_query != query_original:
                        variations.append(new_query)

    return _deduplicate(variations)


def expand_obligation_query(query: str) -> List[str]:
    """Specifically expand queries about legal obligations/responsibilities.

    Uses surgical replacement of obligation patterns to generate queries
    that match the legal text format in EUR-LEX documents.
    """
    synonym_variations = expand_query(query)
    query_original = query.strip()
    query_lower = query_original.lower()

    # Strip leading question words / commands
    core_query = query_lower
    for prefix in ["what are the ", "what is the ", "list the ", "explain the ",
                    "tell me about the ", "tell me about ", "describe the "]:
        if core_query.startswith(prefix):
            core_query = core_query[len(prefix):]
            break

    for pattern in [r"^(what are|what is|what's|how to|how do|how does|how can)\s+"]:
        core_query = re.sub(pattern, "", core_query, count=1).strip()

    # Detect and transform obligation patterns
    obligation_variants = []

    # Pattern 1: "[obligation_noun] of [actor] under [topic]"
    # e.g., "responsibilities of employers under pay transparency"
    ob_noun_pattern = r"(responsibilities|obligations|duties|requirements)\s+of\s+(\w+(?:\s+\w+){0,3})\s+(under|for|in|pursuant|according|following)"
    m = re.search(ob_noun_pattern, core_query)
    if m:
        ob_noun = m.group(1)
        actor = m.group(2)
        preposition = m.group(3)
        rest = core_query[m.end():].strip()

        # Map original obligation noun to legal alternatives
        noun_variants = []
        if ob_noun in ("responsibilities", "obligations", "duties"):
            noun_variants = ["obligations", "duties", "requirements"]
        elif ob_noun == "requirements":
            noun_variants = ["requirements", "obligations", "conditions"]

        # Map actor to legal alternatives
        actor_variants = [actor]
        if "employer" in actor:
            actor_variants = ["employers", "undertakings", "companies"]
        elif "company" in actor or "companies" in actor:
            actor_variants = ["undertakings", "companies", "employers"]
        elif "undertaking" in actor or "undertakings" in actor:
            actor_variants = ["undertakings", "employers", "companies"]

        for nv in noun_variants:
            for av in actor_variants:
                new_q = f"{nv} of {av} {preposition} {rest}"
                obligation_variants.append(new_q)

    # Pattern 2: "[actor] [obligation_verb] [topic]"
    # e.g., "employers must comply with pay transparency"
    actor_verb_pattern = r"(\w+(?:\s+\w+){0,2})\s+(must|shall|required to|obliged to|have to|has to)\s+(comply|report|disclos|ensure|meet)\s+(.*)"
    m = re.search(actor_verb_pattern, core_query)
    if m:
        actor = m.group(1)
        verb = m.group(2)
        action = m.group(3)
        rest = m.group(4).strip()

        obligation_variants.append(f"obligations of {actor} to {action} {rest}")
        obligation_variants.append(f"duties of {actor} regarding {action} {rest}")
        obligation_variants.append(f"requirements for {actor} to {action} {rest}")

    # Pattern 3: Generic obligation prefix (if no specific pattern matched)
    if not obligation_variants:
        obligation_prefixes = [
            "obligations of employers under",
            "duties of undertakings under",
            "requirements for employers under",
            "employer obligations under",
            "compliance requirements under",
        ]
        for prefix in obligation_prefixes:
            obligation_variants.append(f"{prefix} {core_query}")

    # Combine: synonym variants first, then obligation-specific variants
    combined = list(synonym_variations)
    for ov in obligation_variants:
        if ov not in combined:
            combined.append(ov)

    return _deduplicate(combined)


def _replace_insensitive(text: str, old: str, new: str) -> str:
    """Case-insensitive replacement of old with new in text."""
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, text, count=1)


def _deduplicate(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in items:
        item_lower = item.lower().strip()
        if item_lower not in seen:
            seen.add(item_lower)
            result.append(item)
    return result


class AutoExpander:
    """Update query expansion dictionaries based on observed failure patterns.

    Records terms from failed queries and suggests new synonym pairs
    to improve future search recall.
    """

    EXPANSION_FILE = "data/auto_expansions.json"

    def __init__(self):
        self.expansions: Dict[str, List[str]] = {}
        self._load()

    def _load(self):
        import json, os
        if os.path.exists(self.EXPANSION_FILE):
            with open(self.EXPANSION_FILE) as f:
                self.expansions = json.load(f)

    def _save(self):
        import json, os
        os.makedirs(os.path.dirname(self.EXPANSION_FILE) or ".", exist_ok=True)
        with open(self.EXPANSION_FILE, "w") as f:
            json.dump(self.expansions, f, indent=2)

    def record_failure(self, query: str, validation_reason: str):
        """Record a failed query for potential expansion."""
        import re
        words = set(re.findall(r"[a-z]{3,}", query.lower()))
        for w in words:
            if w not in self.expansions:
                self.expansions[w] = []
        self._save()

    def get_auto_expansions(self) -> Dict[str, List[str]]:
        """Get auto-learned expansions (merged with LEGAL_SYNONYMS at query time)."""
        return dict(self.expansions)
