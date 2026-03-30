# 🇩🇪 德國簽證 RAG - 部署指南

> [!NOTE]
> **部署狀態**:
> - **本地 Docker 環境**: ✅ **已完全驗證**。保證開箱即用。
> - **Zeabur 部署**: 🚀 **推薦** 用於生產/演示（目前正在整合中）。
> - **GCP Cloud Run**: 🏛️ **架構參考**。為企業級無伺服器架構提供詳細藍圖。

---

## 目錄

- [前置條件](#前置條件)
- [1. 本地開發 (已驗證)](#1-本地開發-已驗證)
- [2. Zeabur 部署 (生產演示)](#2-zeabur-部署-生產演示)
- [3. GCP Cloud Run (雲端原生參考)](#3-gcp-cloud-run-雲端原生參考)
- [4. 生產環境檢查清單](#4-生產環境檢查清單)
- [5. 故障排除與支援](#5-故障排除與支援)

---

## 前置條件

在部署前，請確保已安裝以下工具：

- **Docker**（v24+）與 **Docker Compose**（v2+）— 本地及容器化部署必要。
- **Node.js**（v18+）與 **npm** — 在本地執行 React 前端必要。
- **Python**（v3.12+）— 在 Docker 外執行資料導入 CLI 必要。
- 有效的 **OpenAI API 金鑰**（或 Azure OpenAI 憑證）。

---

## 1. 本地開發 (已驗證)

這是在您自己的機器上最快速運行完整 RAG 技術棧（API + 向量資料庫 + 快取 + MLflow）的方式。

### 快速開始

```bash
# 1. 設置環境
cp .env.example .env
# 編輯 .env 並填入您的 OPENAI_API_KEY（以及可選的 RERANKER_API_KEY）

# 2. 啟動後端服務（API on :8080、Qdrant on :6333、Redis on :6379、MLflow on :5000）
docker-compose up -d

# 3. 驗證 API 健康狀態（連接埠為 8080，非 8000）
curl -H "X-API-Key: dev-key-12345" http://localhost:8080/v1/health

# 4. 導入數據（CLI）
python -m src.ingestion.cli ingest --auto-discover
```

> [!IMPORTANT]
> API 對外暴露的連接埠為 **8080**，而非 8000。所有 API 呼叫請使用 `http://localhost:8080`。

### 啟動前端（可選）

React 前端位於 `frontend/` 目錄。在本地執行方式如下：

```bash
cd frontend
npm install
npm run dev
# 前端網址：http://localhost:5173
```

請確認 `.env` 中已設置 `ALLOWED_ORIGINS=http://localhost:5173`（`.env.example` 預設已包含此設定）。

### 可選：Azure OpenAI

如需使用 Azure OpenAI 取代標準 OpenAI，請在 `.env` 中設置：

```env
USE_AZURE_OPENAI=true
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_LLM_DEPLOYMENT=gpt-4o-mini
```

### 可選：Ollama（本地 LLM）

若要使用本地執行的 Ollama 模型取代 OpenAI，請在 `docker-compose.yml` 中取消 `ollama` 服務的註解，並設置：

```env
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

### 服務連接埠摘要

| 服務     | 連接埠 | 說明                       |
|----------|--------|----------------------------|
| API      | 8080   | FastAPI RAG 後端           |
| Qdrant   | 6333   | 向量資料庫（REST）         |
| Qdrant   | 6334   | 向量資料庫（gRPC）         |
| Redis    | 6379   | 語義快取                   |
| MLflow   | 5000   | 實驗追蹤（Web UI）         |
| Frontend | 5173   | React UI（開發伺服器）     |

---

## 2. Zeabur 部署 (生產演示)

推薦使用 [Zeabur](https://zeabur.com/) 來託管「火力展示」平台，因其具備無縫的 GitHub 整合與 Docker 支援。

### 第一步：創建新服務

1. 將您的 GitHub 存儲庫連接到 Zeabur。
2. 選擇 `german-visa-rag` 存儲庫。
3. Zeabur 將自動檢測 `Dockerfile` 並開始部署。

### 第二步：配置環境變數

在 Zeabur 儀表板中設置以下變數：

**必要項目：**

| 變數 | 說明 |
|---|---|
| `OPENAI_API_KEY` | 您的 OpenAI API 金鑰 |
| `QDRANT_URL` | Qdrant Cloud 實例的 URL |
| `QDRANT_API_KEY` | Qdrant API 金鑰 |
| `QDRANT_COLLECTION_NAME` | 集合名稱（例如 `german-visa-docs`） |
| `REDIS_URL` | Redis 實例的 URL（用於語義快取） |
| `API_KEY` | API 安全金鑰（通過 `X-API-Key` 標頭傳遞） |
| `ENVIRONMENT` | `production` |

**生產環境建議項目：**

| 變數 | 說明 |
|---|---|
| `ALLOWED_ORIGINS` | 允許的前端來源，以逗號分隔（例如 `https://your-frontend.zeabur.app`） |
| `SQLITE_DB_PATH` | SQLite 狀態資料庫的持久路徑（例如 `/data/state.db`） |
| `RERANKER_API_TYPE` | 重排器後端：`mock`、`cohere` 或 `jina`（預設：`mock`） |
| `RERANKER_API_KEY` | Cohere 或 Jina 重排器的 API 金鑰（使用 `mock` 時留空） |
| `MLFLOW_TRACKING_URI` | MLflow 伺服器 URL（不設置則停用實驗追蹤） |
| `LOG_LEVEL` | 生產環境建議使用 `INFO` 或 `WARNING` |

**可選項目（Azure OpenAI）：**

| 變數 | 說明 |
|---|---|
| `USE_AZURE_OPENAI` | `true` 表示使用 Azure OpenAI 取代標準 OpenAI |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI 金鑰 |
| `AZURE_OPENAI_ENDPOINT` | Azure 端點 URL |
| `AZURE_LLM_DEPLOYMENT` | LLM 的 Azure 部署名稱 |
| `AZURE_EMBEDDING_DEPLOYMENT` | 嵌入模型的 Azure 部署名稱 |

關於 Qdrant Cloud 的設置，請參閱 [`infra/qdrant_cloud_setup.md`](../infra/qdrant_cloud_setup.md)。

### 第三步：部署導入作業

您可以通過 Zeabur 的 **Cron Job** 功能觸發導入 CLI，或使用相同映像創建一個獨立的部署，並將 CMD 覆蓋為：

```
python -m src.ingestion.cli ingest --auto-discover
```

### 第四步：部署前端

`frontend/` 目錄是一個獨立的 React 應用程式。請在 Zeabur 上部署為獨立服務：

1. 創建一個指向相同存儲庫的新服務，但將根目錄設置為 `frontend/`。
2. Zeabur 將自動識別為 Node.js/Vite 專案。
3. 設置環境變數 `VITE_API_URL` 為您的後端 API URL（例如 `https://your-api.zeabur.app`）。
4. 確保後端的 `ALLOWED_ORIGINS` 包含您的前端 URL。

---

## 3. GCP Cloud Run (雲端原生參考)

本節作為在 Google Cloud Platform 上部署高可用性、無伺服器 RAG 架構的技術展示。

### 前置條件

- 已安裝 `gcloud` CLI 並完成身份驗證（`gcloud auth login`）
- 已安裝 Docker（用於構建映像）
- 目標 GCP 專案已啟用 Cloud Run、Secret Manager 和 Artifact Registry API

### 架構藍圖

- **Web API**: 部署為 `Cloud Run Service`（自動擴展，無狀態）。
- **ETL 爬蟲任務**: 部署為 `Cloud Run Job`（防止長時間爬取任務發生 CPU 限制/Throttling）。
- **金鑰管理**: 與 `GCP Secret Manager` 整合。
- **向量資料庫**: 使用 Qdrant Cloud（請參閱 [`infra/qdrant_cloud_setup.md`](../infra/qdrant_cloud_setup.md)）。

### 關鍵命令（僅供參考）

```bash
# 部署 API 服務
# -e: 環境（production/staging）
# -p: GCP 專案 ID
# -r: 區域（例如 europe-west1）
./scripts/deploy.sh -e production -p your-project-id -r europe-west1

# 手動觸發導入作業
gcloud run jobs execute german-visa-rag-job-prod
```

如需深入了解 GCP 部署腳本，請參閱 `infra/` 和 `scripts/` 目錄。完整的 Cloud Run 服務配置位於 `infra/gcp_cloud_run_deploy.yaml`。

---

## 4. 生產環境檢查清單

- [ ] **金鑰安全**: 沒有硬編碼的 API 金鑰；所有金鑰均通過環境變數或 Secret Manager 注入。
- [ ] **連接埠確認**: API 可通過 8080 連接埠存取（非 8000）。
- [ ] **向量資料庫連接**: 已驗證與 Qdrant Cloud 的連接。
- [ ] **語義快取**: Redis 實例可連通（通過 `DEBUG` 日誌確認）。
- [ ] **CORS 設定**: `ALLOWED_ORIGINS` 已設置為前端生產環境的 URL。
- [ ] **前端**: React 應用程式已部署，且 `VITE_API_URL` 指向後端。
- [ ] **健康檢查**: 使用有效的 `X-API-Key` 呼叫 `GET /v1/health` 回傳 200。
- [ ] **引用來源**: 前端顯示 API 返回的 `metadata` 中的來源連結。
- [ ] **爬蟲頻率限制**: 爬蟲已配置為對官方政府網站保持禮貌。
- [ ] **MLflow**: 在生產環境中已進行存取控制或停用（不設置 `MLFLOW_TRACKING_URI`）。
- [ ] **重排器**: 已配置 `RERANKER_API_TYPE`（演示用 `mock`，生產品質建議 `cohere`/`jina`）。

---

## 5. 故障排除與支援

### 常見問題

1. **OOM（記憶體不足）**: 如果運行重排器（Reranker）或查詢轉換器（Query Transformer），請確保容器至少有 2 GB 記憶體。

2. **Qdrant 連接超時**: 檢查 Qdrant Cloud IP 白名單是否允許您的部署出口 IP（或設置 VPC 對等連接）。

3. **無效的 API 金鑰**: 驗證 `X-API-Key` 標頭是否與 `API_KEY` 環境變數匹配。

4. **CORS 錯誤（前端無法連接 API）**: 確保後端的 `ALLOWED_ORIGINS` 包含完整的前端來源（協議 + 主機 + 連接埠，例如 `https://your-app.zeabur.app`）。

5. **連接埠已被佔用**: 如果 8080、6333 或 6379 連接埠已被使用，請修改 `docker-compose.yml` 中的主機連接埠（例如將 `"8080:8080"` 改為 `"18080:8080"`）。

6. **重啟後數據消失**: 確認 Docker 磁碟區（`qdrant_data`、`redis_data`）未在重啟之間被清除。`docker-compose down` **不會**刪除磁碟區，但 `docker-compose down -v` **會**。

7. **導入 CLI 失敗**: 加上 `--log-level DEBUG` 以獲取詳細輸出：
   ```bash
   python -m src.ingestion.cli ingest --auto-discover --log-level DEBUG
   ```

### 支援

如遇技術問題，請至 [GitHub Issues](https://github.com/yourusername/german-visa-rag/issues) 回報，或查閱 `logs/` 目錄中的應用程式日誌。
