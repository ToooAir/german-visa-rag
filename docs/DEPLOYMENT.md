# 🇩🇪 German Visa RAG - Deployment Guide

This project is designed for **Stateless** and **Serverless** environments. To ensure system stability, the production architecture is split into:
1. **Web API**: Deployed as a `Cloud Run Service`, handling real-time Q&A from front-end users.
2. **ETL Crawler Task**: Deployed as a `Cloud Run Job`, partnered with Cloud Scheduler to run periodic web scraping tasks, avoiding Serverless CPU Throttling.

---

## Table of Contents

- [Local Development](#local-development)
- [Docker Build](#docker-build)
- [GCP Cloud Run Deployment](#gcp-cloud-run-deployment)
- [Production Checklist](#production-checklist)
- [Troubleshooting](#troubleshooting)
- [Monitoring and Logging](#monitoring-and-logging)
- [Rollback](#rollback)
- [Cost Optimization](#cost-optimization)
- [Support](#support)

---

## Local Development

### Quick Start

```bash
# 1. Clone and Setup
git clone https://github.com/yourusername/german-visa-rag.git
cd german-visa-rag

# 2. Copy environment variables
cp .env.example .env

# 3. Edit .env and fill in OPENAI_API_KEY
nano .env

# 4. Start all services (API, Qdrant, Redis, MLflow)
docker-compose up -d

# 5. Verify API Status
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/v1/health

# 6. Manually trigger CLI crawler (scraping URLs from config)
python -m src.ingestion.cli ingest
```

### Development Workflow

```bash
# Enter the container for development
docker-compose exec api bash

# Run tests and coverage
pytest tests/ -v --cov=src

# Run Lint
black src/ tests/
ruff check src/

# Run Evaluation
python -m eval.ragas_evaluator eval/eval_dataset.json
```

---

## Docker Build

### Multi-Stage Build

This project uses a Multi-Stage Dockerfile to optimize the final image. **Note: The Web API and CLI crawler share the same Image**, and the run mode is determined via the Command at runtime, which follows best practices.

```dockerfile
# Stage 1: builder (contains all build dependencies)
FROM python:3.11-slim as builder

# Stage 2: runner (contains only runtime dependencies)
FROM python:3.11-slim as runner
COPY --from=builder /opt/venv /opt/venv
```

---

## GCP Cloud Run Deployment

### Prerequisites

1. **GCP Account** and Project ID
2. **gcloud CLI** installed and configured
3. **Qdrant Cloud** instance (Vector Database)
4. **Redis Instance** (for LLM semantic caching)
5. **Secret Manager permissions**

### Step 1: Set up GCP Environment

```bash
export PROJECT_ID="your-gcp-project"
export REGION="europe-west1"

gcloud auth login
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### Step 2: Create Secret Manager Secrets

For security reasons, do not hardcode secrets in environment variables:

```bash
# OpenAI API Key
echo -n "sk-your-api-key" | gcloud secrets create openai-api-key --data-file=-

# Qdrant Configuration
echo -n "https://your-qdrant-instance.qdrant.io" | gcloud secrets create qdrant-url --data-file=-
echo -n "your-qdrant-api-key" | gcloud secrets create qdrant-api-key --data-file=-

# Redis URL (Semantic Cache)
echo -n "redis://your-redis-instance:6379" | gcloud secrets create redis-url --data-file=-

# API System Key
echo -n "your-secure-api-key" | gcloud secrets create api-key --data-file=-
```

### Step 3: Deploy to Cloud Run

#### Option A: Using Deployment Script (Auto deploy Service + Job, Recommended)

```bash
chmod +x scripts/deploy.sh

# Deploy to Staging
./scripts/deploy.sh -e staging -p $PROJECT_ID -r $REGION

# Deploy to Production
./scripts/deploy.sh -e production -p $PROJECT_ID -r $REGION
```

#### Option B: Manual Deployment

If you want to understand the underlying commands, please execute them sequentially:

**1. Deploy Web API (Service)**
```bash
gcloud run deploy german-visa-rag-api-prod \
  --image=gcr.io/$PROJECT_ID/german-visa-rag:latest \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=4Gi \
  --cpu=4 \
  --timeout=600 \
  --set-env-vars=ENVIRONMENT=production,USE_OLLAMA=false \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
  --set-secrets=QDRANT_URL=qdrant-url:latest \
  --set-secrets=QDRANT_API_KEY=qdrant-api-key:latest \
  --set-secrets=REDIS_URL=redis-url:latest \
  --set-secrets=API_KEY=api-key:latest
```

**2. Deploy ETL Crawler (Job)**
```bash
gcloud run jobs deploy german-visa-rag-job-prod \
  --image=gcr.io/$PROJECT_ID/german-visa-rag:latest \
  --region=$REGION \
  --command="python" \
  --args="-m,src.ingestion.cli,ingest" \
  --memory=2Gi \
  --cpu=2 \
  --task-timeout=3600 \
  --set-env-vars=ENVIRONMENT=production,USE_OLLAMA=false \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
  --set-secrets=QDRANT_URL=qdrant-url:latest \
  --set-secrets=QDRANT_API_KEY=qdrant-api-key:latest
```

**3. Set Schedule (Cloud Scheduler)**
```bash
# Set crawler to trigger automatically every Monday at 2 AM
gcloud scheduler jobs create http german-visa-ingestion-scheduler \
  --schedule="0 2 * * 1" \
  --location=$REGION \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/german-visa-rag-job-prod:run" \
  --http-method=POST \
  --oauth-service-account-email="YOUR_COMPUTE_SERVICE_ACCOUNT@developer.gserviceaccount.com"
```

---

## Production Checklist

### Pre-deployment
- [ ] All keys configured in Secret Manager.
- [ ] Qdrant Cloud and Redis instances created and connection tested.
- [ ] Test suite passed (`pytest tests/ -v`).

### Post-deployment
- [ ] Health check endpoint returns 200 (`/health`).
- [ ] **Cache Test**: Send the same query twice, verify that the second response latency drops significantly (Redis Cache Hit).
- [ ] **Job Test**: Verify in Cloud Logging that manual Job execution is successful and has no Errors.
- [ ] SSE Streaming outputs correctly.

---

## Troubleshooting

### 1. Failed to Connect to Qdrant
```bash
# Check Qdrant Cloud connection
curl -H "api-key: your-api-key" https://your-qdrant-instance.qdrant.io/health
```

### 2. Secret Manager Secrets Inaccessible
```bash
# Grant Secret Accessor role to the Service Account
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

### 3. Redis Connection Timeout
If using GCP Memorystore, Cloud Run must be configured with a **Serverless VPC Access Connector** to access the internal IP.
```bash
gcloud run deploy ... --vpc-egress=all-traffic --network=default
```

### 4. Crawler Job Interrupted / Out of Memory
If scraping too many webpages:
- Increase Job timeout limit: `--task-timeout=3600` (1 hour).
- Increase memory allocation: `--memory=4Gi`.

### 5. Ollama Fails to Start on Cloud
For production environments, **it is strongly recommended to disable Ollama** (`USE_OLLAMA=false`). Running LLMs in a CPU-only environment will cause severe latency and OOM issues. The fallback mechanism is only for Local development.

---

## Monitoring and Logging

### View Logs

```bash
# View real-time logs for the API service
gcloud run logs read german-visa-rag-api-prod --limit=100 --follow

# View Crawler Job logs
gcloud run logs read german-visa-rag-job-prod --limit=50 | grep ERROR
```

### Monitoring Metrics
Visit GCP Console > Monitoring > Dashboards, and pay attention to the following metrics:
- Request latency (p50, p95, p99)
- Error rate
- CPU / Memory utilization

---

## Rollback

If there are issues with the deployment, you can rollback the API service to the previous version:

```bash
# List revisions
gcloud run revisions list --service=german-visa-rag-api-prod --region=$REGION

# Route traffic to previous version
gcloud run services update-traffic german-visa-rag-api-prod \
  --to-revisions=PREVIOUS_REVISION_ID=100 \
  --region=$REGION
```

---

## Cost Optimization

### Cloud Run
- **Auto-scaling**: Set `maxScale=10` to scale automatically based on demand.
- **Request Timeout**: Set a reasonable timeout to avoid wasting resources.

### Qdrant Cloud
- **Instance Size**: The Free Tier (1GB) is more than enough to handle tens of thousands of Chunks for this project.

### OpenAI API and Redis
- **Model Selection**: Uses `gpt-4o-mini` by default to significantly reduce costs.
- **Semantic Caching**: **This is the core of cost reduction!** Caching repeated queries via Redis bypassing OpenAI API calls completely, saving massive amounts in Token costs.

---

## Support

Running into issues?
1. Check the [Troubleshooting](#troubleshooting) section
2. Check Cloud Run and Job logs
3. Submit a GitHub Issue
