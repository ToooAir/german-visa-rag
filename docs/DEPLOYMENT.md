# 🇩🇪 German Visa RAG - Deployment Guide

> [!NOTE]
> **Deployment Status**:
> - **Local Docker Environment**: ✅ **Fully Verified**. Guaranteed to work out of the box.
> - **Zeabur Deployment**: 🚀 **Recommended** for production/demo (Currently being integrated).
> - **GCP Cloud Run**: 🏛️ **Architectural Reference**. Detailed blueprint for enterprise-grade serverless infra.

---

## Table of Contents

- [1. Local Development (Verified)](#1-local-development-verified)
- [2. Zeabur Deployment (Production Demo)](#2-zeabur-deployment-production-demo)
- [3. GCP Cloud Run (Cloud-Native Reference)](#3-gcp-cloud-run-cloud-native-reference)
- [4. Production Checklist](#4-production-checklist)
- [5. Troubleshooting & Support](#5-troubleshooting--support)

---

## 1. Local Development (Verified)

This is the fastest way to run the entire RAG stack (API + Vector DB + Cache) on your own machine.

### Quick Start
```bash
# 1. Setup
cp .env.example .env
# Edit .env and insert your OPENAI_API_KEY

# 2. Spin up Services
docker-compose up -d

# 3. Verify API
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/v1/health

# 4. Ingest Data (CLI)
python -m src.ingestion.cli ingest --auto-discover
```

---

## 2. Zeabur Deployment (Production Demo)

[Zeabur](https://zeabur.com/) is recommended for hosting the "Firepower" demonstration due to its seamless GitHub integration and Docker support.

### Step 1: Create New Service
1. Connect your GitHub repository to Zeabur.
2. Select the `german-visa-rag` repository.
3. Zeabur will automatically detect the `Dockerfile` and start the deployment.

### Step 2: Configure Environment Variables
Set the following variables in the Zeabur dashboard:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `QDRANT_URL`: URL to your Qdrant Cloud instance (or a Zeabur-hosted Qdrant).
- `QDRANT_API_KEY`: Your Qdrant API key.
- `REDIS_URL`: URL to your Redis instance (for Semantic Caching).
- `API_KEY`: A secure key for your API (X-API-Key header).
- `ENVIRONMENT`: `production`

### Step 3: Deployment of Ingestion Job
- **Scheduled Tasks**: You can trigger the ingestion CLI via Zeabur's **Cron Job** feature or by creating a separate deployment for the crawler using the same image but overriding the CMD to: `python -m src.ingestion.cli ingest --auto-discover`.

---

## 3. GCP Cloud Run (Cloud-Native Reference)

This section serves as a technical showcase for deploying a high-availability, serverless RAG architecture on Google Cloud Platform.

### Architectural Blueprint
- **Web API**: Deployed as a `Cloud Run Service` (Auto-scaling, Stateless).
- **ETL Crawler Task**: Deployed as a `Cloud Run Job` (Prevents CPU Throttling during long scraping tasks).
- **Secrets Management**: Integrated with `GCP Secret Manager`.

### Key Commands (Reference Only)
```bash
# Deploy API Service
./scripts/deploy.sh -e production -p your-project-id -r europe-west1

# Manual Job Trigger
gcloud run jobs execute german-visa-rag-job-prod
```
*For a deep dive into the GCP deployment scripts, please refer to the `infra/` or `scripts/` directories.*

---

## 4. Production Checklist

- [ ] **Secret Safety**: No API keys are hardcoded; all are injected via ENV or Secret Manager.
- [ ] **Vector DB Connection**: Connection to Qdrant Cloud verified.
- [ ] **Semantic Cache**: Redis instance is reachable (Verify with `DEBUG` logs).
- [ ] **Citations**: Ensure the frontend displays source links from the `metadata` returned by the API.
- [ ] **Rate Limiting**: Crawler configured to be polite to official gov websites.

---

## 5. Troubleshooting & Support

### Common Issues
1. **OOM (Out of Memory)**: Ensure the container has at least 2GB of RAM if running the Reranker or Query Transformer.
2. **Qdrant Connection Timeout**: Check if Qdrant Cloud Whitelist allows your deployment's IP (or set up VPC peering).
3. **Invalid API Key**: Verify `X-API-Key` header matches the `API_KEY` environment variable.

### Support
For technical issues, please check the [GitHub Issues](https://github.com/yourusername/german-visa-rag/issues) or consult the system logs.
Issue
