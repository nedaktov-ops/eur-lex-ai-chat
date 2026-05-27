# Deployment Guide

This guide covers deploying the EUR-Lex AI Chat backend to Render or Vercel, plus the frontend to Vercel.

## Backend Deployment

### Option 1: HuggingFace Spaces (Recommended)

The backend is designed for HuggingFace Spaces with the `cpu-basic` hardware (512MB RAM).

#### Steps

1. **Create a new Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Name: `eurlex-chat-api` (or your preferred name)
   - Select "Docker" as the SDK
   - Set hardware to "cpu-basic" (512MB RAM, free)

2. **Push your code**

```bash
# Clone and push to HF Space
git remote add hf https://huggingface.co/spaces/nedaktovops/eurlex-chat-api
git push hf main --force
```

3. **Set environment variables** in the Space "Settings" → "Variables":

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Your Groq API key |
| `HF_TOKEN` | Yes | HuggingFace token with write access to your dataset |
| `INDEX_SUFFIX` | No | `_eurlex` for EURLEX-BERT embeddings (default: MiniLM) |
| `HF_USERNAME` | No | Override default HF username (default: `NedAktovOps`) |
| `HF_DATASET` | No | Dataset name for index storage (default: `eurlex-chat-data`) |

4. **First run will auto-download the index** (approx 400MB). This may take a few minutes.

5. **Monitor logs** from the Space "Logs" tab to ensure startup completes successfully.

### Option 2: Render

Render offers more control but requires manual setup.

#### Steps

1. **Create a Web Service**

```bash
# In Render dashboard, create new "Web Service"
# Connect your GitHub repository
# Runtime: Docker
# Plan: Free (or paid for more RAM)
```

2. **Environment Variables**

Set in Render "Environment" section:

```bash
GROQ_API_KEY=your_key_here
HF_TOKEN=your_hf_token_here
PORT=8000
# Optional:
INDEX_SUFFIX=_eurlex
HF_USERNAME=NedAktovOps
HF_DATASET=eurlex-chat-data
```

3. **Dockerfile**

The project already includes a `Dockerfile`. Render will build it automatically.

4. **Health Check**

Render expects `/health` endpoint. The API provides it:

```bash
curl https://your-service.onrender.com/health
# Returns: {"status": "ok", "vectors": 305957}
```

5. **Deploy**

Push to GitHub and Render will auto-deploy. First deploy may time out due to index download (400MB). Consider using a paid instance with more RAM or pre-warm the index.

### Option 3: Vercel (with Python)

Vercel's Python support is limited but possible with properly configured `vercel.json`.

1. **Create `vercel.json` in project root**:

```json
{
  "functions": {
    "api/*.py": {
      "runtime": "python3.12",
      "maxDuration": 30
    }
  },
  "crons": [
    {
      "path": "/api/cron/rebuild-index",
      "schedule": "0 2 * * *"
    }
  ]
}
```

2. **Note:** Vercel has a 10-second timeout for serverless functions. The LLM call may exceed this. Consider:

   - Using a separate backend host (Render/HF Spaces) for the API
   - Or moving to Vercel Pro with longer timeouts

**Frontend deployment to Vercel is covered below.**

## Frontend Deployment

### Vercel (Recommended)

1. **Build and Deploy**

```bash
cd frontend
npm install
npm run build
npx vercel --prod
```

2. **Configure Environment**

Create `.env` in `frontend/`:

```bash
VITE_API_URL=https://your-backend-url.com
```

Or set in Vercel dashboard under "Environment Variables".

3. **Custom Domain (Optional)**

In Vercel dashboard:
- Go to "Domains"
- Add your domain
- Update `VITE_API_URL` if needed

## Environment Variables Reference

### Required

| Variable | Description | Where to get |
|----------|-------------|--------------|
| `GROQ_API_KEY` | Groq API key for LLM inference | [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | HuggingFace token for index download/backup | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEX_SUFFIX` | `""` (empty) | Suffix for index files: `""` uses MiniLM, `"_eurlex"` uses EURLEX-BERT |
| `HF_USERNAME` | `NedAktovOps` | HuggingFace username/organization |
| `HF_DATASET` | `eurlex-chat-data` | Dataset name containing the index |
| `FROM_DATE` | `2004-01-01` | For index rebuild: earliest document date |
| `PORT` | `8000` | Server port (Render sets this automatically) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Override Groq model name |
| `VITE_API_URL` | (none) | Frontend: override backend API URL |

## Automated Index Updates with Cron

### Using GitHub Actions (Already Configured)

The repo includes `.github/workflows/backup.yml` which runs daily at 2 AM and:

1. Triggers a backup of the current index
2. Optionally rebuilds with EURLEX-BERT if `INDEX_SUFFIX=_eurlex`

To modify:

Edit `.github/workflows/backup.yml`:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:     # Allow manual trigger
```

### Using Render Cron Jobs

Render supports cron via separate "Cron Job" service:

1. Create a new "Cron Job" in Render dashboard
2. Command: `bash scripts/run_build.sh`
3. Schedule: `0 2 * * *` (daily at 2 AM)
4. Set environment variables (same as web service)
5. Ensure the Cron Job has access to the same HF_TOKEN

### Using HuggingFace Spaces (Static Site Alternative)

HF Spaces does not support cron. Use GitHub Actions to trigger rebuild:

```yaml
name: Daily rebuild
on:
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:

jobs:
  rebuild:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Trigger rebuild via API
        run: |
          curl -X POST https://huggingface.co/api/spaces/nedaktovops/eurlex-chat-api/restart \
            -H "Authorization: Bearer ${{ secrets.HF_TOKEN }}"
```

**Note**: This restarts the Space, triggering `startup` which can download a fresh index.

## Monitoring & Observability

### Health Checks

- `GET /health` — Returns `{"status": "ok", "vectors": 305957}` if index is loaded
- `GET /stats` — Returns index statistics: file sizes, last updated, memory usage

Check from your monitoring system:

```bash
curl -f https://your-backend.com/health || echo "Service down"
```

### Logs

#### HuggingFace Spaces

- View logs in the Space "Logs" tab
- Or via API:

```bash
curl -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/spaces/nedaktovops/eurlex-chat-api/logs
```

#### Render

- Logs available in Render dashboard under "Logs"
- Stream with `render logs <service-name>`

The API emits structured JSON logs for each pipeline stage:

```json
{
  "timestamp": "2026-05-26T02:14:24.115Z",
  "level": "INFO",
  "request_id": "req-abc123",
  "stage": "search_performed",
  "duration_ms": 142,
  "top_scores": [0.68, 0.62, 0.58],
  "results_count": 10
}
```

### Performance Metrics

Monitor these key numbers:

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Search latency (p95) | < 500ms | 500-1000ms | > 1000ms |
| Answer generation (p95) | < 2000ms | 2000-4000ms | > 4000ms |
| Validation pass rate | > 85% | 70-85% | < 70% |
| Memory usage (HF Spaces) | < 400MB | 400-500MB | > 512MB (OOM risk) |

For RAGAS metrics (faithfulness, context recall), run `scripts/evaluate.py` periodically.

### Alerting Recommendations

Set up alerts for:

1. **Health check failures** — `GET /health` returns non-ok status
2. **High error rate** — 5xx responses > 1% over 5 minutes
3. **Memory pressure** — approaching 512MB limit on HF Spaces
4. **Quality degradation** — RAGAS faithfulness < 0.7 or recall < 0.6

Example using UptimeRobot or similar:

```
https://your-backend.com/health
Check interval: 1 minute
Timeout: 10 seconds
Alert on: HTTP != 200 OR response doesn't contain "ok"
```

### Log Analysis

Run the feedback analyzer on pipeline logs:

```bash
cat logs/pipeline.log | python3 scripts/feedback_analyzer.py
```

Outputs:
- Validation pass rate
- Latency distribution (avg, p50, p95)
- Confidence distribution (high/medium/low)
- Intent distribution
- Answer length stats
- Citation counts

## Troubleshooting

### Out of Memory on HF Spaces

The `cpu-basic` tier has 512MB RAM. If you see OOM errors:

1. Reduce `top_k` in search (edit `app/main.py`)
2. Use MiniLM instead of EURLEX-BERT (don't set `INDEX_SUFFIX=_eurlex`)
3. Enable swap (not possible on free tier) — upgrade to "cpu-small"

### Slow First Request

The first request after a cold start loads the index (~400MB). This can take 30-60 seconds. Consider:

- Upgrading to a paid instance with more CPU/RAM
- Keeping the service warm with a cron ping every 5 minutes

### Missing Models or Dependencies

If you see `ImportError` for `chunkweaver` or `ragas`:

```bash
# The Dockerfile installs dependencies from app/requirements.txt
# Ensure it includes:
#   chunkweaver>=0.1.0
#   ragas>=0.2.0
#   rank-bm25>=0.2.2
#   eurlxp>=0.1.0
```

Rebuild the Docker image and redeploy.

### Index Not Found

Ensure `HF_TOKEN` has `read` access to the dataset. The index is stored at:

```
hf://datasets/NedAktovOps/eurlex-chat-data/
  ├── index.faiss
  ├── index_eurlex.faiss
  ├── chunks.db
  └── build_meta.json
```

If files are missing, run a rebuild:

```bash
HF_TOKEN=your_token bash scripts/run_build.sh
```

### Groq API Errors

Check:
- `GROQ_API_KEY` is valid and not expired
- You haven't exceeded rate limits (check Groq dashboard)
- Model name in `groq_api_call()` exists and is accessible

### Reranker Causing Slowdowns

The cross-encoder reranker adds ~100-200ms per query. If this is too slow:

1. Reduce `top_k` from 10 to 5 in the reranker call
2. Or disable reranking by commenting out the reranker section in `app/main.py` (Stage 4)

## Security Considerations

- Never commit `.env` files. They are gitignored.
- Rotate `GROQ_API_KEY` and `HF_TOKEN` periodically.
- Use read-only tokens when possible for deployment.
- The API has no authentication — consider adding API key validation if exposed publicly.

## Maintenance

### Daily Tasks (Automated)

- Index backup: `.github/workflows/backup.yml`
- Feedback analysis: `.github/workflows/feedback-analysis.yml`

### Weekly Tasks

- Review logs for error patterns
- Check RAGAS metrics (`scripts/evaluate.py`)
- Monitor storage growth on HuggingFace Hub

### Monthly Tasks

- Update dependencies (chunkweaver, ragas, sentence-transformers)
- Re-evaluate coverage against SPARQL (`scripts/benchmark_coverage.py`)
- Test recovery from latest checkpoint if needed

## Rollback Procedures

If a new index deployment causes issues:

1. **On HuggingFace Spaces** — Restore from backup:

```bash
# View available backups
ls data/backup-*/

# Restore latest
cp -r data/backup- latest/* data/
git add data/
git commit -m "Rollback to previous index"
git push hf main --force
```

2. **Using checkpoints** — Restore from a saved checkpoint:

```bash
python3 scripts/checkpoint_restore.py --list
python3 scripts/checkpoint_restore.py --id ckpt-20260523-204006
```

3. **Render** — Redeploy previous Git commit:

```bash
git revert <bad-commit>
git push origin main
# Render auto-deploys previous version
```

## Support

- Issues: https://github.com/nedaktov-ops/eur-lex-ai-chat/issues
- Documentation: See `docs/` directory
- Project metadata: `implementation-plan.md`, `STRATEGY.md`
