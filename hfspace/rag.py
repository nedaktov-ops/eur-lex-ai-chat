"""Build prompts and call Groq API for RAG."""

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a legal AI assistant specialized in EU law. You help users understand EU legislation by answering their questions based on provided context from EUR-Lex documents.

Guidelines:
1. Answer based ONLY on the provided context. If the context doesn't contain enough information, say so.
2. Always cite the specific EUR-Lex document(s) you used with their CELEX numbers.
3. When citing articles, include the article number and CELEX number.
4. Keep answers clear and accessible — explain legal concepts in plain language.
5. If the user asks in a non-English language, respond in that language.
6. Do not make up legal citations or references. Only cite what's in the context.
7. Be honest about limitations — if you're unsure, say so."""


def build_prompt(query, context_chunks):
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = f"[{i+1}] CELEX {chunk['celex']}"
        if chunk.get("article"):
            source += f", Article {chunk['article']}"
        context_parts.append(f"Context {i+1} ({source}):\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""Here are relevant excerpts from EU law documents:

{context_str}

Based on the above legal texts, please answer the following question:

{query}"""
    return prompt


def call_groq(prompt, max_retries=3):
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
        "max_tokens": 2048,
    }

    for attempt in range(max_retries):
        try:
            r = httpx.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            if r.status_code == 429:
                logger.warning(f"Groq rate limited (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue
            logger.error(f"Groq API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None

    return None


def extract_citations(text):
    celex_pattern = r"CELEX\s+(\d{2,4}[A-Z0-9]+(?:\([A-Z0-9]+\))?(?:\([0-9]+\))?)"
    return re.findall(celex_pattern, text)


def answer_question(query, context_chunks):
    prompt = build_prompt(query, context_chunks)
    answer = call_groq(prompt)

    if not answer:
        return {
            "answer": "Sorry, I couldn't generate an answer right now. Please try again.",
            "citations": [],
        }

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
