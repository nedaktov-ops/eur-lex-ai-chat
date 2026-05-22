---
title: EUR-Lex AI Chat API
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# EUR-Lex AI Chat API

FastAPI backend for EUR-Lex AI Chat. Accepts questions about EU law and returns answers with citations.

## Endpoints

- `GET /health` — Status check
- `GET /refresh` — Check for index updates
- `POST /chat` — Ask a question

## Environment Variables

- `HF_USERNAME` — HF Hub username (default: NedAktovOps)
- `HF_DATASET` — Dataset name (default: eurlex-chat-data)
- `HF_TOKEN` — HF read token
- `GROQ_API_KEY` — Groq API key for LLM
