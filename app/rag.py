"""Build prompts and call Groq API for RAG."""

import json
import logging
import os
import re
import time as time_module

import httpx

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger("eurlex-chat.pipeline")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def log_pipeline_stage(stage, request_id, data, level="INFO"):
    """Log a structured pipeline stage event (independent copy for rag.py)."""
    log_entry = {
        "timestamp": time_module.strftime("%Y-%m-%dT%H:%M:%S.", time_module.gmtime()) + f"{time_module.time() % 1:.6f}"[2:8] + "Z",
        "request_id": request_id,
        "stage": stage,
        "data": data,
    }
    log_level = getattr(logging, level.upper(), logging.INFO)
    rag_logger.log(log_level, json.dumps(log_entry))

SYSTEM_PROMPT = """You are a legal AI assistant specialized in EU law. You help users understand EU legislation by answering their questions based on provided context from EUR-Lex documents.

CRITICAL: Your answer MUST include CELEX numbers INLINE for each source you reference. Every factual claim must be attributed to a specific CELEX document. Use the format: CELEX 32023L0970.

Guidelines:
1. Answer based ONLY on the provided context. If the context doesn't contain enough information, say so.
2. Always cite the specific EUR-Lex document(s) you used with their CELEX numbers INLINE in your answer text (e.g., "Under Directive 2023/970 (CELEX 32023L0970), employers must...").
3. When citing articles, include the article number and CELEX number (e.g., "Article 5 of Directive 2023/970 (CELEX 32023L0970)").
4. Keep answers clear and accessible — explain legal concepts in plain language.
5. If the user asks in a non-English language, respond in that language.
6. Do not make up legal citations or references. Only cite what's in the context.
7. Be honest about limitations — if you're unsure, say so.
8. When discussing obligations, distinguish between mandatory requirements ('shall') and permissions ('may').
9. Synthesize information from multiple sources when relevant.

OUTPUT FORMAT:
- Write a concise, informative answer.
- Include CELEX numbers INLINE for every source you use, like: "Under Directive 2023/970 (CELEX 32023L0970), employers must..."
- Your answer MUST contain at least 2 CELEX numbers from the provided context.

Example:
User: What are the transparency obligations for employers under Directive 2023/970?
Assistant: Under Directive 2023/970 (CELEX 32023L0970), employers must provide salary information to prospective employees. Additionally, Directive 2018/1972 (CELEX 32018L1972) contains complementary provisions on pay transparency reporting."""

ENSURE_CITATION_PROMPT = """
CRITICAL: Your previous answer did NOT include enough CELEX number citations. This is a HARD REQUIREMENT.
- Your answer MUST include at least 2 CELEX numbers INLINE in the text, each prefixed by 'CELEX ' (e.g., CELEX 32023L0970).
- Every factual claim must be attributed to a specific CELEX document.
- If you use information from a provided context chunk, cite its CELEX number inline.
- Do not use generic references like "the directive" or "the regulation" without the CELEX number.
- Example format: "Under Directive 2023/970 (CELEX 32023L0970), employers must...\nAdditionally, Directive 2018/1972 (CELEX 32018L1972) requires..." """


def build_prompt(query, context_chunks, classification=None, extra_system_notes=None):
    """Build an enhanced prompt with legal discourse awareness.

    Args:
        query: The user's query string.
        context_chunks: List of search result dicts with text, celex, article, etc.
        classification: Optional dict from question_classifier.

    Returns:
        The assembled prompt string.
    """
    from .relation_extractor import extract_legal_relations, summarize_relations

    # Part 1: Intent-specific system instructions
    system_notes = []
    if extra_system_notes:
        system_notes.append(extra_system_notes)
    if classification:
        if classification.get("obligation_seeking"):
            system_notes.append(
                "The user is asking about legal obligations or responsibilities. "
                "Focus on identifying and explaining specific duties imposed by the law, "
                "citing which articles create each obligation."
            )
        elif classification.get("legal_intent") == "definition":
            system_notes.append(
                "The user is asking for a definition. "
                "Provide the precise legal definition from the directive, "
                "citing the relevant article."
            )

    # Part 2: Context chunks with discourse metadata and relation extraction
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = f"[{i+1}] CELEX {chunk['celex']}"
        if chunk.get("article"):
            source += f", Article {chunk['article']}"

        # Add discourse marker for obligation-heavy chunks
        discourse_boost = chunk.get("discourse_boost", 1.0)
        if discourse_boost > 1.2:
            source += " [CONTAINS OBLIGATION LANGUAGE]"

        # Run relation extraction per chunk
        chunk_relations = extract_legal_relations(chunk["text"])
        relation_summary = summarize_relations(chunk_relations)

        chunk_text = (
            f"Context {i+1} ({source}):\n{chunk['text']}\n"
            f"--- Legal structure: {relation_summary}"
        )
        context_parts.append(chunk_text)

    context_str = "\n\n---\n\n".join(context_parts)

    # Assemble final prompt
    system_str = "\n".join(system_notes) if system_notes else ""

    # Build the user message without duplicating SYSTEM_PROMPT (it's in the system role)
    parts = []
    if system_str:
        parts.append(system_str)
    parts.append(f"""Relevant excerpts from EU law documents:

{context_str}

Based on the above legal texts, please answer the following question:

{query}

Remember to:
1. Answer directly using the provided legal texts
2. Include CELEX numbers (e.g., 32023L0970) INLINE in your answer for every source you reference
3. Distinguish between mandatory obligations ('shall') and permissions ('may')
4. If the texts don't fully answer the question, clearly state what information is missing
5. Synthesize information from multiple sources when relevant
6. Every source you cite must have its CELEX number in the answer text (at least 2 distinct sources)""")

    prompt = "\n\n".join(parts)
    return prompt


def call_groq(prompt, max_retries=3):
    """Call Groq API. Returns dict with answer + metadata or None on failure."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set")
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }

    for attempt in range(max_retries):
        start_time = time_module.time()
        try:
            r = httpx.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            duration_ms = (time_module.time() - start_time) * 1000
            if r.status_code == 429:
                logger.warning(f"Groq rate limited (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time_module.sleep(2 ** attempt)
                    continue
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage", {})
            return {
                "answer": data["choices"][0]["message"]["content"],
                "model": GROQ_MODEL,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "duration_ms": round(duration_ms, 2),
                "success": True,
            }
        except httpx.HTTPStatusError as e:
            duration_ms = (time_module.time() - start_time) * 1000
            if e.response.status_code == 429 and attempt < max_retries - 1:
                time_module.sleep(2 ** attempt)
                continue
            logger.error(f"Groq API error: {e}")
            return None
        except Exception as e:
            duration_ms = (time_module.time() - start_time) * 1000
            logger.error(f"Groq API error: {e}")
            return None

    return None


def extract_citations(text):
    # Match raw CELEX numbers (e.g., 32023L0970) anywhere in the text.
    # CELEX format: starts with '3', then 4 digits, then an uppercase letter, then 4 digits.
    pattern = r"3\d{4}[A-Z]\d{4}"
    return re.findall(pattern, text)


def answer_question(query, context_chunks, request_id=None, classification=None,
                    retry_with_citation_emphasis=False):
    """Run the full RAG pipeline: build prompt, call LLM, extract citations.

    Args:
        query: The user's query string.
        context_chunks: List of context chunk dicts from search.
        request_id: Optional request ID for logging.
        classification: Optional classification result from question_classifier.

    Returns:
        Dict with answer, citations, and sources.
    """
    extra_notes = None
    if retry_with_citation_emphasis:
        extra_notes = ENSURE_CITATION_PROMPT + "\n\nThis is your SECOND attempt. The first answer failed because it did not include enough CELEX citations. You MUST include at least 2 CELEX numbers inline this time."
    prompt = build_prompt(query, context_chunks, classification=classification,
                          extra_system_notes=extra_notes)
    prompt_length = len(prompt)

    # If enhanced prompt is too long, fall back to shorter version with fewer chunks
    if prompt_length > 40000 and len(context_chunks) > 5:
        logger.warning(f"Prompt too long ({prompt_length} chars), reducing to 5 chunks")
        prompt = build_prompt(query, context_chunks[:5], classification=classification,
                              extra_system_notes=extra_notes)
        prompt_length = len(prompt)

    # Log prompt built (include classification if available)
    log_data = {
        "prompt_length": prompt_length,
        "context_chunks_count": len(context_chunks),
    }
    if classification:
        log_data["legal_intent"] = classification.get("legal_intent")
        log_data["obligation_seeking"] = classification.get("obligation_seeking")
    log_pipeline_stage("prompt_built", request_id, log_data)

    # Call LLM with timing
    llm_result = call_groq(prompt)

    if llm_result is None or not llm_result.get("answer"):
        # Log failed LLM call
        log_pipeline_stage("llm_call", request_id, {
            "model": GROQ_MODEL,
            "prompt_tokens": None,
            "completion_tokens": None,
            "duration_ms": None,
            "success": False,
        }, level="ERROR")

        log_pipeline_stage("answer_generated", request_id, {
            "answer_length": 0,
            "citations_count": 0,
            "sources_count": 0,
            "validation_passed": None,
        }, level="WARNING")

        return {
            "answer": "Sorry, I couldn't generate an answer right now. Please try again.",
            "citations": [],
            "sources": [],
        }

    answer = llm_result["answer"]

    # Log successful LLM call
    log_pipeline_stage("llm_call", request_id, {
        "model": llm_result.get("model", GROQ_MODEL),
        "prompt_tokens": llm_result.get("prompt_tokens"),
        "completion_tokens": llm_result.get("completion_tokens"),
        "total_tokens": (llm_result.get("prompt_tokens") or 0) + (llm_result.get("completion_tokens") or 0),
        "duration_ms": llm_result.get("duration_ms"),
        "success": True,
    })

    citations = extract_citations(answer)

    source_citations = []
    seen = set()
    for chunk in context_chunks:
        if chunk["celex"] not in seen:
            seen.add(chunk["celex"])
            source_citations.append({
                "celex": chunk["celex"],
                "title": chunk["title"],
                "article": chunk.get("article"),
                "score": chunk["score"],
            })

    return {
        "answer": answer,
        "citations": citations,
        "sources": source_citations,
    }
