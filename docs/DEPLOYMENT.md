# 🇩🇪 German Visa RAG - Deployment Guide

> [!NOTE]
> **Deployment Status**:
> - **Local Docker Environment**: ✅ **Fully Verified**. Guaranteed to work out of the box.
> - **Zeabur Deployment**: 🚀 **Recommended** for production/demo (Currently being integrated).
> - **GCP Cloud Run**: 🏛️ **Architectural Reference**. Detailed blueprint for enterprise-grade serverless infra.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Local Development (Verified)](#1-local-development-verified)
- [2. Zeabur Deployment (Production Demo)](#2-zeabur-deployment-production-demo)
- [3. GCP Cloud Run (Cloud-Native Reference)](#3-gcp-cloud-run-cloud-native-reference)
- [4. Production Checklist](#4-production-checklist)
- [5. Troubleshooting & Support](#5-troubleshooting--support)

---

## Prerequisites

Before deploying, ensure you have the following installed:

- **Docker** (v24+) and **Docker Compose** (v2+) — required for local and container-based deployments.
- **Node.js** (v18+) and **npm** — required to run the React frontend locally.
- **Python** (v3.12+) — required to run the ingestion CLI outside of Docker.
- A valid **OpenAI API key** (or Azure OpenAI credentials).

---

## 1. Local Development (Verified)

This is the fastest way to run the entire RAG stack (API + Vector DB + Cache + MLflow) on your own machine.

### Quick Start

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env and insert your OPENAI_API_KEY (and optionally RERANKER_API_KEY)

# 2. Spin up backend services (API on :8080, Qdrant on :6333, Redis on :6379, MLflow on :5000)
docker-compose up -d

# 3. Verify API health (port is 8080, not 8000)
curl -H "X-API-Key: dev-key-12345" http://localhost:8080/v1/health

# 4. Ingest data (CLI)
python -m src.ingestion.cli ingest --auto-discover
```

> [!IMPORTANT]
> The API is exposed on **port 8080**, not 8000. Use `http://localhost:8080` for all API calls.

### Start the Frontend (Optional)

The React frontend is located in the `frontend/` directory. To run it locally:

```bash
cd frontend
npm install
npm run dev
# Frontend available at http://localhost:5173
```

Make sure `ALLOWED_ORIGINS=http://localhost:5173` is set in your `.env` (it is set by default in `.env.example`).

### Optional: Azure OpenAI

If you want to use Azure OpenAI instead of standard OpenAI, set the following in `.env`:

```env
USE_AZURE_OPENAI=true
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_LLM_DEPLOYMENT=gpt-4o-mini
```

### Optional: Ollama (Local LLM)

To use a locally-running Ollama model instead of OpenAI, uncomment the `ollama` service in `docker-compose.yml` and set:

```env
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

### Service Ports Summary

| Service  | Port  | Description              |
|----------|-------|--------------------------|
| API      | 8080  | FastAPI RAG backend      |
| Qdrant   | 6333  | Vector DB (REST)         |
| Qdrant   | 6334  | Vector DB (gRPC)         |
| Redis    | 6379  | Semantic cache           |
| MLflow   | 5000  | Experiment tracking (UI) |
| Frontend | 5173  | React UI (dev server)    |

---

## 2. Zeabur Deployment (Production Demo)

[Zeabur](https://zeabur.com/) is recommended for hosting the "Firepower" demonstration due to its seamless GitHub integration and Docker support.

### Step 1: Create New Service

1. Connect your GitHub repository to Zeabur.
2. Select the `german-visa-rag` repository.
3. Zeabur will automatically detect the `Dockerfile` and start the deployment.

### Step 2: Configure Environment Variables

Set the following variables in the Zeabur dashboard:

**Required:**

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `QDRANT_URL` | URL to your Qdrant Cloud instance |
| `QDRANT_API_KEY` | Your Qdrant API key |
| `QDRANT_COLLECTION_NAME` | Collection name (e.g. `german-visa-docs`) |
| `REDIS_URL` | URL to your Redis instance (for Semantic Caching) |
| `API_KEY` | A secure key for your API (sent via `X-API-Key` header) |
| `ENVIRONMENT` | `production` |

**Recommended for production:**

| Variable | Description |
|---|---|
| `ALLOWED_ORIGINS` | Comma-separated list of allowed frontend origins (e.g. `https://your-frontend.zeabur.app`) |
| `SQLITE_DB_PATH` | Persistent path for SQLite state store (e.g. `/data/state.db`) |
| `RERANKER_API_TYPE` | Reranker backend: `mock`, `cohere`, or `jina` (default: `mock`) |
| `RERANKER_API_KEY` | API key for Cohere or Jina reranker (leave empty for `mock`) |
| `MLFLOW_TRACKING_URI` | MLflow server URL (omit to disable experiment tracking) |
| `LOG_LEVEL` | `INFO` or `WARNING` for production |

**Optional (Azure OpenAI):**

| Variable | Description |
|---|---|
| `USE_AZURE_OPENAI` | `true` to use Azure OpenAI instead of standard OpenAI |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint URL |
| `AZURE_LLM_DEPLOYMENT` | Azure deployment name for LLM |
| `AZURE_EMBEDDING_DEPLOYMENT` | Azure deployment name for embeddings |

For Qdrant Cloud setup, follow the guide at [`infra/qdrant_cloud_setup.md`](../infra/qdrant_cloud_setup.md).

### Step 3: Deploy Ingestion Job

You can trigger the ingestion CLI via Zeabur's **Cron Job** feature, or by creating a separate deployment using the same image but overriding the CMD to:

```
python -m src.ingestion.cli ingest --auto-discover
```

### Step 4: Deploy the Frontend

The `frontend/` directory is a standalone React app. Deploy it as a separate Zeabur service:

1. Create a new service pointing to the same repository, but set the root directory to `frontend/`.
2. Zeabur will detect it as a Node.js/Vite project.
3. Set the environment variable `VITE_API_URL` to your backend API URL (e.g. `https://your-api.zeabur.app`).
4. Ensure the backend's `ALLOWED_ORIGINS` includes your frontend URL.

---

## 3. GCP Cloud Run (Cloud-Native Reference)

This section serves as a technical showcase for deploying a high-availability, serverless RAG architecture on Google Cloud Platform.

### Prerequisites

- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- Docker installed for building images
- Target GCP project with Cloud Run, Secret Manager, and Artifact Registry APIs enabled

### Architectural Blueprint

- **Web API**: Deployed as a `Cloud Run Service` (Auto-scaling, Stateless).
- **ETL Crawler Task**: Deployed as a `Cloud Run Job` (Prevents CPU Throttling during long scraping tasks).
- **Secrets Management**: Integrated with `GCP Secret Manager`.
- **Vector DB**: Use Qdrant Cloud (see [`infra/qdrant_cloud_setup.md`](../infra/qdrant_cloud_setup.md)).

### Key Commands (Reference Only)

```bash
# Deploy API Service
# -e: environment (production/staging)
# -p: GCP project ID
# -r: region (e.g. europe-west1)
./scripts/deploy.sh -e production -p your-project-id -r europe-west1

# Manual Job Trigger (ingestion)
gcloud run jobs execute german-visa-rag-job-prod
```

For a deep dive into the GCP deployment scripts, refer to the `infra/` and `scripts/` directories. The full Cloud Run service configuration is at `infra/gcp_cloud_run_deploy.yaml`.

---

## 4. Production Checklist

- [ ] **Secret Safety**: No API keys are hardcoded; all injected via ENV or Secret Manager.
- [ ] **Port**: API is accessible on port 8080 (not 8000).
- [ ] **Vector DB Connection**: Connection to Qdrant Cloud verified.
- [ ] **Semantic Cache**: Redis instance is reachable (verify with `DEBUG` logs).
- [ ] **CORS**: `ALLOWED_ORIGINS` set to your frontend's production URL(s).
- [ ] **Frontend**: React app deployed and `VITE_API_URL` points to backend.
- [ ] **Health Check**: `GET /v1/health` returns 200 with a valid `X-API-Key`.
- [ ] **Citations**: Frontend displays source links from `metadata` returned by the API.
- [ ] **Rate Limiting**: Crawler configured to be polite to official gov websites.
- [ ] **MLflow**: Either secured behind auth or disabled (`MLFLOW_TRACKING_URI` unset) in production.
- [ ] **Reranker**: `RERANKER_API_TYPE` configured (`mock` for demo, `cohere`/`jina` for production quality).

---

## 5. Troubleshooting & Support

### Common Issues

1. **OOM (Out of Memory)**: Ensure the container has at least 2 GB of RAM if running the Reranker or Query Transformer.

2. **Qdrant Connection Timeout**: Check if Qdrant Cloud IP whitelist allows your deployment's egress IP (or set up VPC peering).

3. **Invalid API Key**: Verify the `X-API-Key` header matches the `API_KEY` environment variable.

4. **CORS Error (frontend can't reach API)**: Ensure `ALLOWED_ORIGINS` in your backend includes the exact frontend origin (scheme + host + port, e.g. `https://your-app.zeabur.app`).

5. **Port Already in Use**: If port 8080, 6333, or 6379 is already occupied, change the host port in `docker-compose.yml` (e.g. `"18080:8080"`).

6. **Data Not Persisting After Restart**: Confirm that Docker volumes (`qdrant_data`, `redis_data`) are not being pruned between restarts (`docker-compose down` does **not** delete volumes; `docker-compose down -v` does).

7. **Ingestion CLI Fails**: Run with `--log-level DEBUG` for verbose output:
   ```bash
   python -m src.ingestion.cli ingest --auto-discover --log-level DEBUG
   ```

### Support

For technical issues, please open a [GitHub Issue](https://github.com/ToooAir/german-visa-rag/issues) or consult the application logs in the `logs/` directory.
