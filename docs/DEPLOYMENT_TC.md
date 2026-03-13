
# 🇩🇪 German Visa RAG - 部署指南 (Deployment Guide)

本專案專為 **無狀態 (Stateless)** 與 **Serverless** 環境設計。為確保系統穩定性，生產環境架構已拆分為：
1. **Web API**: 部署為 `Cloud Run Service`，處理前端使用者的即時問答。
2. **ETL 爬蟲任務**: 部署為 `Cloud Run Job`，搭配 Cloud Scheduler 進行定時排程抓取，避免 Serverless CPU Throttling。

---

## 目錄

- [本地開發](#本地開發)
- [Docker 構建](#docker-構建)
- [GCP Cloud Run 部署](#gcp-cloud-run-部署)
- [生產環境檢查清單](#生產環境檢查清單)
- [故障排除](#故障排除)
- [監控與日誌](#監控與日誌)
- [回滾](#回滾)
- [成本優化](#成本優化)

---

## 本地開發

### 快速啟動

```bash
# 1. Clone 和設置
git clone https://github.com/yourusername/german-visa-rag.git
cd german-visa-rag

# 2. 複製環境變數
cp .env.example .env

# 3. 編輯 .env，填入 OPENAI_API_KEY
nano .env

# 4. 啟動所有服務 (API, Qdrant, Redis, MLflow)
docker-compose up -d

# 5. 驗證 API 狀態
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/v1/health

# 6. 手動觸發 CLI 爬蟲 (抓取設定檔網址)
python -m src.ingestion.cli ingest
```

### 開發工作流

```bash
# 進入容器開發
docker-compose exec api bash

# 運行測試與覆蓋率
pytest tests/ -v --cov=src

# 運行 Lint
black src/ tests/
ruff check src/

# 運行評測
python -m eval.ragas_evaluator eval/eval_dataset.json
```

---

## Docker 構建

### Multi-Stage Build

本項目使用 Multi-Stage Dockerfile 優化最終映像。**注意：Web API 與 CLI 爬蟲共用同一個 Image**，透過啟動時的指令 (Command) 來決定運行模式，這符合最佳實踐。

```dockerfile
# Stage 1: builder (包含所有構建依賴)
FROM python:3.11-slim as builder

# Stage 2: runner (只包含運行時依賴)
FROM python:3.11-slim as runner
COPY --from=builder /opt/venv /opt/venv
```

---

## GCP Cloud Run 部署

### 先決條件

1. **GCP 帳戶** 與 Project ID
2. **gcloud CLI** 已安裝並配置
3. **Qdrant Cloud** 實例（向量數據庫）
4. **Redis 實例**（用於 LLM 語意快取）
5. **Secret Manager 權限**

### 步驟 1: 設置 GCP 環境

```bash
export PROJECT_ID="your-gcp-project"
export REGION="europe-west1"

gcloud auth login
gcloud config set project $PROJECT_ID

# 啟用必要 API
gcloud services enable run.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### 步驟 2: 建立 Secret Manager 密鑰

為了安全性，請勿將密鑰寫死在環境變數中：

```bash
# OpenAI API Key
echo -n "sk-your-api-key" | gcloud secrets create openai-api-key --data-file=-

# Qdrant 配置
echo -n "https://your-qdrant-instance.qdrant.io" | gcloud secrets create qdrant-url --data-file=-
echo -n "your-qdrant-api-key" | gcloud secrets create qdrant-api-key --data-file=-

# Redis URL (Semantic Cache)
echo -n "redis://your-redis-instance:6379" | gcloud secrets create redis-url --data-file=-

# API 系統金鑰
echo -n "your-secure-api-key" | gcloud secrets create api-key --data-file=-
```

### 步驟 3: 部署到 Cloud Run

#### 選項 A: 使用部署腳本（自動部署 Service + Job，推薦）

```bash
chmod +x scripts/deploy.sh

# 部署到 Staging
./scripts/deploy.sh -e staging -p $PROJECT_ID -r $REGION

# 部署到 Production
./scripts/deploy.sh -e production -p $PROJECT_ID -r $REGION
```

#### 選項 B: 手動部署

如果你想了解底層指令，請依次執行：

**1. 部署 Web API (Service)**
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

**2. 部署 ETL 爬蟲 (Job)**
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

**3. 設定排程 (Cloud Scheduler)**
```bash
# 設定每週一凌晨 2 點自動觸發爬蟲
gcloud scheduler jobs create http german-visa-ingestion-scheduler \
  --schedule="0 2 * * 1" \
  --location=$REGION \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/german-visa-rag-job-prod:run" \
  --http-method=POST \
  --oauth-service-account-email="YOUR_COMPUTE_SERVICE_ACCOUNT@developer.gserviceaccount.com"
```

---

## 生產環境檢查清單

### 部署前
- [ ] 所有密鑰在 Secret Manager 中配置。
- [ ] Qdrant Cloud 與 Redis 實例已創建並測試連線。
- [ ] 測試套件已通過（`pytest tests/ -v`）。

### 部署後
- [ ] 健康檢查端點返回 200 (`/health`)。
- [ ] **快取測試**：發送相同查詢兩次，確認第二次延遲大幅降低 (Redis Cache Hit)。
- [ ] **Job 測試**：在 Cloud Logging 中確認 Job 手動執行成功且無 Error。
- [ ] SSE Streaming 輸出正常。

---

## 故障排除

### 1. 連接到 Qdrant 失敗
```bash
# 檢查 Qdrant Cloud 連接
curl -H "api-key: your-api-key" https://your-qdrant-instance.qdrant.io/health
```

### 2. Secret Manager 密鑰無法訪問
```bash
# 授予 Service Account Secret Accessor 角色
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

### 3. Redis 連線超時 (Timeout)
如果使用 GCP Memorystore，Cloud Run 必須配置 **Serverless VPC Access Connector** 才能訪問內網 IP。
```bash
gcloud run deploy ... --vpc-egress=all-traffic --network=default
```

### 4. 爬蟲 Job 被中斷 / 內存不足
如果爬取網頁過多：
- 增加 Job 的超時限制：`--task-timeout=3600` (1小時)。
- 增加記憶體配置：`--memory=4Gi`。

### 5. Ollama 在雲端啟動失敗
生產環境**強烈建議關閉 Ollama** (`USE_OLLAMA=false`)。純 CPU 環境跑 LLM 會導致嚴重的延遲與 OOM。備援機制僅限 Local 開發。

---

## 監控與日誌

### 查看日誌

```bash
# 查看 API 服務即時日誌
gcloud run logs read german-visa-rag-api-prod --limit=100 --follow

# 查看 Crawler Job 日誌
gcloud run logs read german-visa-rag-job-prod --limit=50 | grep ERROR
```

### 監控指標
訪問 GCP Console > Monitoring > Dashboards，關注以下指標：
- Request latency (p50, p95, p99)
- Error rate
- CPU / Memory utilization

---

## 回滾

如果部署出現問題，可以回滾 API 服務到前一版本：

```bash
# 列出修訂版本
gcloud run revisions list --service=german-visa-rag-api-prod --region=$REGION

# 路由流量到前一版本
gcloud run services update-traffic german-visa-rag-api-prod \
  --to-revisions=PREVIOUS_REVISION_ID=100 \
  --region=$REGION
```

---

## 成本優化

### Cloud Run
- **自動擴展**：設置 `maxScale=10`，自動根據需求擴展。
- **請求超時**：設置合理超時，避免浪費資源。

### Qdrant Cloud
- **實例大小**：Free Tier (1GB) 足以應對本專案數萬個 Chunks。

### OpenAI API 與 Redis
- **模型選擇**：預設使用 `gpt-4o-mini` 極大降低成本。
- **Semantic Caching**：**這是降低成本的核心！** 透過 Redis 快取重複查詢，直接繞過 OpenAI API 呼叫，省下大量 Token 費用。

---

## 支持

遇到問題？
1. 查看 [故障排除](#故障排除) 部分
2. 檢查 Cloud Run 與 Job 日誌
3. 提交 GitHub Issue