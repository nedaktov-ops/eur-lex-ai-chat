"""Lightweight legal relation extraction for EU directive text.

Identifies:
- Obligations (shall, must, required to)
- Rights (has the right to, may, entitled to)
- Prohibitions (shall not, prohibited, not permitted)
- Conditions (provided that, subject to, where)
- Legal actors (employer, controller, member state, etc.)
"""

import re

OBLIGATION_PATTERNS = [
    (r"(shall|must)\s+(ensure|take|establish|implement|provide|report|disclose|notify|adopt|designate|maintain)", "positive_obligation"),
    (r"(shall|must)\s+[a-z]+\s+and\s+(shall|must)", "compound_obligation"),
    (r"has\s+(?:the\s+)?duty\s+to", "positive_obligation"),
    (r"is\s+responsible\s+for", "responsibility"),
    (r"is\s+required\s+to", "requirement"),
    (r"shall\s+be\s+(?:responsible|accountable|liable)\s+for", "accountability"),
    (r"(obligation|duty|responsibility)\s+(?:to|of|shall|under)", "obligation_statement"),
]

PROHIBITION_PATTERNS = [
    (r"shall\s+not\s+(permit|allow|use|disclose|process|transfer|discriminate)", "prohibition"),
    (r"prohibited\s+from", "prohibition"),
    (r"not\s+permitted\s+to", "prohibition"),
    (r"may\s+not\s+(use|disclose|process|transfer|discriminate)", "restriction"),
    (r"shall\s+not\s+be\s+(construed|interpreted)\s+as", "non_application"),
    (r"no\s+(person|employer|undertaking|organisation)\s+(shall|may)", "prohibition"),
]

RIGHT_PATTERNS = [
    (r"has\s+(?:the\s+)?right\s+to", "right"),
    (r"is\s+entitled\s+to", "entitlement"),
    (r"may\s+(?:request|receive|access|obtain|submit|appeal)", "right"),
    (r"have\s+the\s+right\s+to", "right"),
    (r"(?:right|entitlement)\s+(?:of|to|under)", "right_statement"),
    (r"shall\s+have\s+the\s+right\s+to", "right"),
]

CONDITION_PATTERNS = [
    (r"provided\s+that", "condition"),
    (r"subject\s+to", "condition"),
    (r"where\s+the\s+(?:employer|controller|processor|undertaking|member\s+state)", "condition"),
    (r"unless\s+", "exception"),
    (r"except\s+where", "exception"),
    (r"in\s+the\s+event\s+that", "condition"),
    (r"in\s+accordance\s+with", "compliance_condition"),
]

ACTOR_PATTERNS = [
    r"(employer|employee|worker|applicant|job\s+seeker)",
    r"(controller|processor|data\s+subject)",
    r"(member\s+state|national\s+authority|competent\s+authority)",
    r"(commission|council|parliament|european\s+parliament)",
    r"(undertaking|company|enterprise|organization)",
]

DEONTIC_PATTERNS = {
    "shall", "must", "required", "obliged", "duty", "duties",
    "responsible", "liability", "sanction", "penalty", "breach",
    "prohibited", "not permitted", "shall ensure", "shall take",
    "shall establish", "shall implement", "shall report",
    "obligation", "obligations", "responsibility", "responsibilities",
    "mandatory", "compliance", "compliant",
}


def extract_legal_relations(text: str) -> dict:
    """Extract legal relations from text chunks.

    Args:
        text: The text content of a chunk.

    Returns:
        Dict with keys: obligations, prohibitions, rights, conditions, actors
    """
    relations = {
        "obligations": [],
        "prohibitions": [],
        "rights": [],
        "conditions": [],
        "actors": [],
    }

    text_lower = text.lower()

    def _extract_with_context(patterns, text_lower, text, context_window=60):
        results = []
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text_lower):
                start = max(0, match.start() - context_window)
                end = min(len(text), match.end() + context_window)
                context = text[start:end]
                results.append({
                    "type": rel_type,
                    "matched_text": match.group(0),
                    "context": context.strip(),
                })
        return results

    relations["obligations"] = _extract_with_context(OBLIGATION_PATTERNS, text_lower, text)
    relations["prohibitions"] = _extract_with_context(PROHIBITION_PATTERNS, text_lower, text, 50)
    relations["rights"] = _extract_with_context(RIGHT_PATTERNS, text_lower, text, 50)
    relations["conditions"] = _extract_with_context(CONDITION_PATTERNS, text_lower, text, 40)

    # Extract actors
    seen_actors = set()
    for pattern in ACTOR_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            actor = match.group(0).strip()
            if actor not in seen_actors:
                seen_actors.add(actor)
                relations["actors"].append(actor)

    return relations


def has_deontic_language(text: str) -> bool:
    """Check if text contains deontic (obligation) language."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in DEONTIC_PATTERNS)


def count_deontic_density(text: str) -> int:
    """Count the number of deontic language occurrences in text."""
    text_lower = text.lower()
    return sum(text_lower.count(kw) for kw in DEONTIC_PATTERNS)


def summarize_relations(relations: dict) -> str:
    """Create a human-readable summary of legal relations for prompt injection.

    Args:
        relations: Output from extract_legal_relations()

    Returns:
        A compact text summary of the legal relations found.
    """
    parts = []

    if relations["obligations"]:
        parts.append(f"OBLIGATIONS ({len(relations['obligations'])} found):")
        seen_types = {}
        for obl in relations["obligations"]:
            t = obl["type"]
            if t not in seen_types:
                seen_types[t] = 0
            seen_types[t] += 1
        for t, count in sorted(seen_types.items(), key=lambda x: -x[1]):
            parts.append(f"  - {t}: {count} instance(s)")

    if relations["prohibitions"]:
        parts.append(f"PROHIBITIONS ({len(relations['prohibitions'])} found):")
        for pro in relations["prohibitions"][:2]:
            ctx = pro["context"][:80]
            parts.append(f"  - {pro['type']}: ...{ctx}...")

    if relations["rights"]:
        parts.append(f"RIGHTS ({len(relations['rights'])} found)")
        for right in relations["rights"][:2]:
            ctx = right["context"][:80]
            parts.append(f"  - {right['type']}: ...{ctx}...")

    if relations["conditions"]:
        parts.append(f"CONDITIONS ({len(relations['conditions'])} found)")

    if relations["actors"]:
        parts.append(f"ACTORS: {', '.join(relations['actors'][:5])}")

    return "\n".join(parts)
