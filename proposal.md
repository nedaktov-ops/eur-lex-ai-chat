# EUR-Lex AI Chat — Project Proposal

**Date:** May 20, 2026
**Goal:** A free, open-source chat interface over EU law — ask questions in plain language, get answers with citations to real EUR-Lex documents. Runs 100% online after one-time build. Zero ongoing cost. Zero manual maintenance. Discoverable via Google/AI search.

**Total cost:** $0.00
**Laptop involvement:** One-time initial build (+ content writing). After that: zero.
**Build target:** 2 weeks

---

## How People Find This Project

Chat apps are invisible to search engines. A React SPA (our original plan) would ship blank HTML to Googlebot — no content, no structure, no ranking. **This project will fail if no one can find it.**

The fix: **Astro framework** instead of React+Vite. Astro ships zero JavaScript by default, producing pure HTML that Googlebot and AI crawlers (GPTBot, ClaudeBot) read instantly. The chat widget is embedded as a React "island" — interactive for users, invisible to crawlers — while the rest of the page is fully indexable content.

### How discovery works

```
Search engine crawler (Googlebot, GPTBot, ClaudeBot)
  │
  ▼
Landing page (pure HTML, zero JS)
  ├── Title: "EUR-Lex AI Chat — Ask EU Law Questions in Plain English"
  ├── H1: Clear, keyword-rich headline
  ├── FAQ section: "What is a CELEX number?", "How to find EU regulations?"
  │   └── JSON-LD FAQPage schema → rich Google snippets
  ├── Blog posts: "What is the GDPR?", "EU AI Act explained"
  │   └── JSON-LD Article schema → rich Google snippets
  ├── SoftwareApplication JSON-LD → recognized as a tool by Google
  └── Sitemap.xml → tells Google every URL on the site
        │
        ▼
  Google indexes all content pages
        │
        ▼
  User searches "EU AI Act requirements" or "what is CELEX number"
        │
        ▼
  Our FAQ page or blog post appears in search results
        │
        ▼
  User clicks, reads, tries the chat widget → converts
```

### Search engine visibility comparison

| Aspect | React SPA (Vite) — OLD | Astro — NEW |
|--------|----------------------|-------------|
| HTML sent to crawler | Blank `<div id="root"></div>` | Full rendered content |
| JavaScript shipped | 80-150KB framework + app code | Zero (unless interacting with chat) |
| Lighthouse score (typical) | 45-65 | 95-100 |
| Core Web Vitals (LCP) | 2.5-4s | 0.5-1.2s |
| Indexable by Googlebot | Needs prerendering workarounds | ✅ Out of the box |
| Indexable by GPTBot/ClaudeBot | Needs SSR | ✅ Out of the box |
| Rich snippets (FAQ, Article) | Difficult (client-side JSON-LD breaks) | ✅ Static JSON-LD in HTML |
| Content pages (blog, FAQ) | Separate build, extra complexity | ✅ Built-in (Markdown, same repo) |

---

## The 6 Free Services That Keep It Alive

### 1. HuggingFace Hub — Data Storage ($0, no time limit)

Stores the vector index and document chunks. Public dataset repo, free storage.

```
Dataset:  your-github-username/eurlex-chat-data
├── vectors.npy       # ~154MB — all pre-computed embeddings
├── chunks.json       # ~50MB — text chunks + metadata
└── last_updated.txt  # timestamp — used for incremental updates
```

### 2. GitHub Actions — Data Pipeline ($0, 2,000 min/month free)

Runs the scraper, chunker, and embedder every day at 06:00 UTC. Uses `uv` (Rust-based pip) for sub-second dependency installs.

```yaml
# .github/workflows/update-index.yml
name: Daily index update
on:
  schedule:
    - cron: "0 6 * * *"
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install eurlxp[sparql] numpy huggingface_hub sentence-transformers[onnx]
      - run: python scripts/update_index.py
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

Each run: ~5-10 min (only 20-50 new/changed docs per day). Monthly: ~300 min of 2,000 free.

### 3. Render — Backend API Server ($0, 512MB RAM)

Runs FastAPI. Loads vectors from HF Hub at startup. Hot-reloads data hourly (checks HF Hub for newer files).

**Always awake:** cron-job.org pings `/health` every 5 minutes. 

**Self-updating:** Same cron pings `/refresh` every 60 minutes → checks HF Hub → hot-reloads if newer data exists. No deploy needed.

### 4. Vercel — Frontend ($0, never sleeps)

Astro site. Static pages (landing, blog, FAQ) are pre-rendered to pure HTML at build time. The chat widget is a React island embedded on the page. Deployed from GitHub — auto-deploys on every push.

Zero server cost. Never sleeps. 100GB bandwidth free.

### 5. cron-job.org — Keep-Alive & Refresh ($0)

Two free cron jobs:

| Job | Interval | URL | Purpose |
|-----|----------|-----|---------|
| Keep-alive | Every 5 min | `GET /health` | Prevents Render sleep |
| Refresh | Every 60 min | `GET /refresh` | Triggers data hot-reload |

### 6. Groq — AI ($0, 1,000 requests/day)

Llama 3.3 70B via API. No credit card. 1,000 requests/day free. Rate-limited per IP to 20/day/user.

---

## SEO Strategy — How We Get Found

### What we installed

| Tool | What it does | Cost | Installed on |
|------|-------------|------|-------------|
| **seofor.dev** | CLI SEO audit tool — crawls your dev server, checks titles/meta/headings/images/links, exports AI prompts to fix issues | Free (MIT, Go binary) | Your laptop (for development) |
| **Google Search Console** | Tracks indexing status, crawl errors, Core Web Vitals, keyword rankings | Free (web-based) | Just a meta tag in the site |
| **Google Rich Results Test** | Validates JSON-LD structured data | Free (web-based) | Used during development |
| **Astro sitemap integration** | Auto-generates sitemap.xml on build | Free (built-in) | Part of the framework |

### What's built into the site

**1. Content pages (blog + FAQ) — the main SEO driver**

People search for EU law questions every day. If our site has pages answering those questions, Google shows them in results. Each page drives traffic to the chat tool.

```
/                          → Landing page (SoftwareApplication schema)
/chat                      → The actual chat widget (React island)
/blog/eu-ai-act-explained  → Blog post (Article schema, drives search traffic)
/blog/gdpr-article-17      → Blog post
/blog/what-is-celex        → Blog post
/faq                       → FAQ page (FAQPage schema → rich Google snippets)
```

Blog posts are written in Markdown. Astro renders them to static HTML at build time. No database, no server, no cost.

**2. JSON-LD structured data** — tells Google exactly what this is

Every page gets machine-readable metadata:

```json
// Landing page — tells Google this is a software tool
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "EUR-Lex AI Chat",
  "applicationCategory": "WebApplication",
  "operatingSystem": "Web",
  "description": "Chat about EU law in plain English. Get answers with citations to real EUR-Lex documents.",
  "url": "https://eurlex-chat.vercel.app",
  "offers": { "@type": "Offer", "price": "0", "priceCurrency": "EUR" }
}

// FAQ page — generates expandable results in Google search
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    { "@type": "Question", "name": "What is a CELEX number?",
      "acceptedAnswer": { "@type": "Answer", "text": "..." } }
  ]
}
```

**3. Sitemap + robots.txt**

`sitemap.xml` is auto-generated by Astro on every build. Tells Google every page and when it was last updated. `robots.txt` allows all crawlers.

**4. Open Graph + Twitter Card tags**

Every page gets `og:title`, `og:description`, `og:image` — so links look good when shared on social media, Discord, Slack.

### SEO content roadmap

| Content | SEO impact | Effort |
|---------|-----------|--------|
| Landing page with clear value prop | High (first impression, schema) | 2 hours |
| FAQ page: 20 common EU law questions | High (FAQ rich snippets in Google) | 3 hours |
| 5 blog posts: GDPR, DMA, DSA, AI Act, CELEX | High (keyword traffic) | 5 hours |
| Blog post each time EU publishes major regulation | Medium (fresh content signal) | 1 hour/month |
| Backlinks from GitHub + open-source directories | High (domain authority) | 1 hour/setup |

**Total SEO content effort:** ~12 hours once, then ~1 hour/month.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE ARCHITECTURE                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  DATA FLOW (fully automated):                                             │
│                                                                           │
│  EUR-Lex Cellar SPARQL (public)                                          │
│       │ GitHub Actions queries daily                                     │
│       ▼                                                                  │
│  EUR-Lex Cellar REST API (public)                                        │
│       │ GitHub Actions downloads HTML                                     │
│       ▼                                                                  │
│  GitHub Actions Runner (free, 7GB RAM, 2-core CPU)                       │
│       │ uv pip install → chunk → embed (ONNX) → merge                     │
│       ▼                                                                  │
│  HuggingFace Hub Dataset (public, free storage)                          │
│       │ vectors.npy + chunks.json                                         │
│       │ ▲                        │                                       │
│       │ │                        ▼                                       │
│       │ │               Render (512MB, free)                             │
│       │ │               ├── startup: download from HF                     │
│       │ │               ├── hourly: check HF for updates                  │
│       │ │               └── live: FastAPI + numpy KNN                     │
│       │ │                        │                                       │
│       │ │                        ▼                                       │
│       │ │              ┌─────────────────┐                               │
│       └────────────────┤  Groq API       │                               │
│                        │  (Llama 3.3 70B)│                               │
│                        └────────┬────────┘                               │
│                                 │                                         │
│  USER FLOW:                     │                                         │
│  [Google Search]                │                                         │
│       │ user finds FAQ/blog post                                          │
│       ▼                                                                   │
│  [Browser] ─► [Vercel: Astro site] ─► [Render: FastAPI API] ─► [Groq]   │
│                Static HTML pages     /chat endpoint         answer        │
│                + React chat island   numpy KNN + RAG                      │
│                                 │                                         │
│  KEEP ALIVE:                    ▼                                         │
│  [cron-job.org] ──► [Render /health] every 5 min                         │
│  [cron-job.org] ──► [Render /refresh] every 60 min                       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Files Layout

```
eur-lex-ai-chat/
├── backend/
│   ├── main.py                 # FastAPI: /chat, /health, /refresh
│   ├── search.py               # numpy KNN search over pre-loaded vectors
│   ├── rag.py                  # Build prompt, call Groq, parse citations
│   ├── data_loader.py          # Download index from HF Hub at startup
│   ├── rate_limit.py           # Per-IP + global rate limiting
│   ├── requirements.txt        # fastapi, uvicorn, numpy, huggingface_hub, httpx
│   └── startup.sh              # Entry point: download data, start uvicorn
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── index.astro         # Landing page (SEO content + chat island)
│   │   │   ├── chat.astro          # Chat page (React island)
│   │   │   ├── faq.astro           # FAQ page (JSON-LD + rich snippets)
│   │   │   └── blog/
│   │   │       ├── index.astro     # Blog listing
│   │   │       └── posts/          # Markdown blog posts
│   │   │           ├── eu-ai-act.md
│   │   │           ├── gdpr-guide.md
│   │   │           └── what-is-celex.md
│   │   ├── components/
│   │   │   ├── ChatWidget.jsx      # React chat island
│   │   │   └── SeoHead.astro      # JSON-LD + OG tags component
│   │   └── layouts/
│   │       └── Base.astro          # Main layout with nav, footer, SEO tags
│   ├── astro.config.mjs
│   ├── tailwind.config.mjs
│   └── package.json
├── scripts/
│   ├── build_index.py          # Laptop: FULL build from scratch (one-time)
│   └── update_index.py         # GitHub Actions: INCREMENTAL update (daily)
├── .github/
│   └── workflows/
│       └── update-index.yml    # GitHub Actions workflow (daily)
├── proposal.md
└── README.md
```

---

## Your Laptop Does (One-Time Build)

| Step | What | Time |
|------|------|------|
| 1 | Run `build_index.py` — downloads all 27K in-force EUR-Lex docs, chunks, embeds | ~3 hours |
| 2 | Create HF dataset + upload vectors.npy + chunks.json | 2 min |
| 3 | Scaffold Astro project, write landing page, FAQ, blog posts | ~12 hours |
| 4 | Push code to GitHub — triggers Vercel + Render deploy | 1 min |
| 5 | Set up cron-job.org (2 jobs: keepalive + refresh) | 2 min |
| 6 | Add site to Google Search Console | 5 min |
| 7 | Run `seofordev audit` to verify SEO health | 2 min |
| **Done.** | **Laptop powers off. Project runs itself forever.** | **~15 hours total** |

After this, everything runs online:
- GitHub Actions scrapes daily updates
- HuggingFace stores the data
- Render serves the API  
- Vercel serves the frontend (with indexable SEO content)
- cron-job keeps it alive
- Groq answers questions
- Google indexes the content pages → users find it naturally

---

## $0 Budget Breakdown

| Service | Purpose | Free Tier Limit | Annual Cost |
|---------|---------|----------------|-------------|
| HuggingFace Hub | Store vectors + chunks | Best-effort public storage | $0 |
| GitHub Actions | Daily scraper + embedder | 2,000 min/month (uses ~300) | $0 |
| Render | FastAPI backend | 512MB RAM, 750 hours/month | $0 |
| Vercel | Astro frontend | 100GB bandwidth, never sleeps | $0 |
| cron-job.org | Keep-alive + refresh | 2 jobs, free unlimited | $0 |
| Groq | AI (Llama 3.3 70B) | 1,000 requests/day | $0 |
| GitHub | Code hosting | Unlimited public repos | $0 |
| **Total** | | | **$0.00** |

---

## What Happens If...

| Scenario | Outcome | Recovery |
|----------|---------|----------|
| Render restarts | Loads vectors from HF Hub at startup (~2 sec) | Automatic |
| GitHub Actions fails today | Old index stays. App keeps working. | Fixes itself tomorrow |
| HuggingFace Hub is down | Render keeps running with last-loaded index | Automatic when HF recovers |
| Groq hits daily limit | Returns "Rate limit reached, try tomorrow" | Automatic next day |
| cron-job.org goes down | Render sleeps after 15 min. First request cold starts (~30 sec) | Next ping wakes it |
| EUR-Lex changes their API | Scraper fails. GitHub Actions sends email alert. | Fix script, push update |
| A regulation is repealed | Removed from index on next daily update | Automatic |
| Google changes SEO rules | Content pages still work. Might lose some rich snippets. | Update JSON-LD if needed |
| No blog posts written yet | FAQ page still gets indexed. Landing page still ranks. | Write posts when you have time |

---

## Build Plan

| Day | What | Deliverable |
|-----|------|-------------|
| 1 | `scripts/build_index.py` — SPARQL → Cellar REST → parse → chunk → embed | Data pipeline script |
| 2 | **Run build_index.py** — full index of 27K docs (~3 hours). Upload to HF Hub. | vectors.npy + chunks.json on HF |
| 3 | **Scaffold Astro frontend** — landing page, JSON-LD, sitemap, layout, Tailwind | Astro project with SEO basics |
| 4 | **Build chat island** — React chat component embedded in Astro. Connect to backend. | Working chat on the site |
| 5 | **Build FastAPI backend** — `/chat`, `/health`, `/refresh`. Load from HF Hub. | Working API |
| 6 | **Write SEO content** — FAQ (20 questions), 3 blog posts. Deploy everything. | Indexable pages live |
| 7 | **GitHub Actions + cron-job + Search Console** — full automation. Laptop off. | Fully autonomous system |

---

## Free SEO Tools Installed

| Tool | Purpose | How to use |
|------|---------|-----------|
| **seofor.dev** | CLI SEO audit | `seo audit run --port 8080` while dev server is running |
| **Google Search Console** | Track indexing, keywords, Core Web Vitals | Add meta tag to site head → verify ownership → monitor |
| **Google Rich Results Test** | Validate JSON-LD | `search.google.com/test/rich-results` — paste URL |
| **Astro sitemap** | Auto-generate sitemap.xml | Built-in integration (`astro add sitemap`) |
| **PageSpeed Insights** | Core Web Vitals score | `pagespeed.web.dev` — paste URL |
| **Google Mobile-Friendly Test** | Mobile SEO check | `search.google.com/test/mobile-friendly` |
