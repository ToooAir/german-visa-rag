# 🇩🇪 German Visa & Chancenkarte RAG API

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF5252.svg?logo=qdrant)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/Redis-Cache-DC382D.svg?logo=redis)](https://redis.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **Advanced RAG (Retrieval-Augmented Generation)** API system designed to answer complex legal and application questions regarding German Visas and the "Chancenkarte" (Opportunity Card). The system natively supports queries in English, German, and Chinese, ensuring all generated answers are strictly grounded in authoritative, official sources with precise citations.

Built with **Production-ready** standards, this project features an automated web ingestion pipeline, canonical state deduplication, Hybrid Search, Cross-Encoder Reranking, LLM-based query transformation, **Redis Semantic Caching**, and a complete CI/CD workflow.

## ✨ Core Features

### 🔍 Advanced RAG Pipeline
- **Query Transformation**: Utilizes a lightweight LLM for intent expansion and spell-checking to solve multi-lingual vector space misalignment.
- **Hybrid Search**: Combines **Dense Vectors** (OpenAI `text-embedding-3-small`) with **Sparse BM25** search via Qdrant.
- **Cross-Encoder Reranking**: Fetches Top-20 candidates and reranks them using a Cross-Encoder API to distill the precise Top-5 chunks.
- **Time-Aware & Authority Weighting**: Prioritizes official government sources and recently fetched documents during retrieval scoring.

### 🚀 Performance & Cost Optimization
- **Semantic Caching**: Integrates Redis to cache LLM responses based on deterministic query hashing. Delivers **~10ms response times** for repeated queries.
- **Enhanced Parent-Child Chunking**: Implements a "Small-to-Big" strategy with **Title Context Injection** and **Noise Removal** (strips images/boilerplate) for 80% cleaner RAG context.

### 🛠️ Engineering Excellence
- **LLM Factory Pattern (Local Fallback)**: Implements dependency inversion. If the OpenAI API key is missing or offline, the system seamlessly falls back to a local **Ollama** model (ideal for local resilience testing).
- **Standalone CLI Ingestion Script**: Decouples the ETL pipeline from the Web API. The provided CLI perfectly aligns with Serverless environments (e.g., GCP Cloud Run Jobs) to prevent CPU throttling during web crawling.
- **OpenAI-Compatible API**: Fully implements the `POST /v1/chat/completions` endpoint with SSE Streaming support.
- **Defensive Programming**: Built-in Prompt Injection detection and a Global Exception Handler.

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        A1["Web Client Chat UI"]
        A2["OpenAI-compatible SDK"]
    end

    subgraph "API Gateway (FastAPI)"
        B1["/v1/chat/completions"]
        B2["/query/ask (RAG specific)"]
        EH["Global Exception Handler"]
    end

    subgraph "Query Processing & Cache"
        C1["Query Transformer"]
        E2[("Redis Semantic Cache")]
    end

    subgraph "Retrieval Pipeline"
        D1["Hybrid Search (Vector + BM25)"]
        D2["Cross-Encoder Reranker"]
        D3["Prompt Builder (+ Safety Check)"]
        F1{{"LLM Factory"}}
        LLM_A["OpenAI"]
        LLM_B["Local Ollama"]
    end

    subgraph "Data Ingestion (CLI / Jobs)"
        G0(("CLI: python -m src.ingestion.cli"))
        G1["Crawler -> HTML to MD"]
        G2["Parent-Child Chunker"]
        G3["Canonical Hash Dedup"]
    end

    subgraph "Storage & Infrastructure"
        E1[("Qdrant (Dense + Sparse)")]
        E3[("SQLite (Ingestion State)")]
    end

    A1 --> B1
    A2 --> B1
    B1 <--> E2
    B1 --> C1 --> D1 --> D2 --> D3
    D3 --> F1
    F1 --> LLM_A
    F1 -. Fallback .-> LLM_B
    D1 <--> E1
    
    G0 --> G1 --> G2 --> G3 --> E1
    G3 <--> E3
```

---

## 🚀 Quick Start (Local Development)

### 1. Setup
```bash
git clone https://github.com/yourusername/german-visa-rag.git
cd german-visa-rag
cp .env.example .env
# Edit .env and insert your OPENAI_API_KEY
```

### 2. Spin Up Services
```bash
docker-compose up -d
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/v1/health
```

### 3. Trigger Data Ingestion (CLI)
Use the dedicated CLI tool to trigger the web crawler and ETL pipeline:
```bash
# Ingest all URLs from config
python -m src.ingestion.cli ingest

# Auto-discover and ingest all pages from defined domains
python -m src.ingestion.cli ingest --auto-discover

# Force re-ingestion and apply new processing logic to existing docs
python -m src.ingestion.cli ingest --auto-discover --force

# Test ingestion on a single URL
python -m src.ingestion.cli ingest --source "https://www.make-it-in-germany.com/en/"
```

---

## 💻 API Usage Example

The API is strictly OpenAI-compatible. You can point the official Python SDK directly to your local instance.

```python
from openai import OpenAI

client = OpenAI(
    api_key="dev-key-12345",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are the requirements for the Chancenkarte?"}],
    stream=True
)

for chunk in response:
    print(chunk.choices.delta.content or "", end="")
```

---

## 🧪 Testing & Evaluation (MLOps)

```bash
docker-compose exec api bash

# 1. Run Tests & Coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# 2. Run Ragas Pipeline Evaluation
python -m eval.ragas_evaluator eval/eval_dataset.json
```

---

## ☁️ Deployment

Designed for stateless deployment on **GCP Cloud Run (API)** and **GCP Cloud Run Jobs (CLI)** backed by **Qdrant Cloud** and **Redis Cloud**.

```bash
./scripts/deploy.sh -e production -p your-gcp-project-id -r europe-west1
```

---

## ⚠️ Disclaimer
**This project is built for technical demonstration purposes (Side Project)**. All answers are generated by AI and **do not constitute legal advice**. Always refer to official announcements from the [Federal Foreign Office](https://www.auswaertiges-amt.de/en) or [Make it in Germany](https://www.make-it-in-germany.com/en/).
