# 🇩🇪 德國簽證 RAG - 部署指南

> [!NOTE]
> **部署狀態**: 
> - **本地 Docker 環境**: ✅ **已完全驗證**。保證開箱即用。
> - **Zeabur 部署**: 🚀 **推薦** 用於生產/演示（目前正在整合中）。
> - **GCP Cloud Run**: 🏛️ **架構參考**。為企業級無伺服器架構提供詳細藍圖。

---

## 目錄

- [1. 本地開發 (已驗證)](#1-本地開發-已驗證)
- [2. Zeabur 部署 (生產演示)](#2-zeabur-部署-生產演示)
- [3. GCP Cloud Run (雲端原生參考)](#3-gcp-cloud-run-雲端原生參考)
- [4. 生產環境檢查清單](#4-生產環境檢查清單)
- [5. 故障排除與支援](#5-故障排除與支援)

---

## 1. 本地開發 (已驗證)

這是最快在您自己的機器上運行完整 RAG 技術棧（API + 向量資料庫 + 快取）的方式。

### 快速開始
```bash
# 1. 設置
cp .env.example .env
# 編輯 .env 並插入您的 OPENAI_API_KEY

# 2. 啟動服務
docker-compose up -d

# 3. 驗證 API
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/v1/health

# 4. 導入數據 (CLI)
python -m src.ingestion.cli ingest --auto-discover
```

---

## 2. Zeabur 部署 (生產演示)

推薦使用 [Zeabur](https://zeabur.com/) 來託管「火力展示」平台，因其具備無縫的 GitHub 整合與 Docker 支援。

### 第一步：創建新服務
1. 將您的 GitHub 存儲庫連接到 Zeabur。
2. 選擇 `german-visa-rag` 存儲庫。
3. Zeabur 將自動檢測 `Dockerfile` 並開始部署。

### 第二步：配置環境變數
在 Zeabur 儀表板中設置以下變數：
- `OPENAI_API_KEY`: 您的 OpenAI API 金鑰。
- `QDRANT_URL`: 您的 Qdrant Cloud 實例 URL（或 Zeabur 託管的 Qdrant）。
- `QDRANT_API_KEY`: 您的 Qdrant API 金鑰。
- `REDIS_URL`: 您的 Redis 實例 URL（用於語義快取）。
- `API_KEY`: 您的 API 安全金鑰（X-API-Key 標頭）。
- `ENVIRONMENT`: `production`

### 第三步：部署導入作業 (Ingestion Job)
- **定時任務**: 您可以通過 Zeabur 的 **Cron Job** 功能觸發導入 CLI，或者使用相同的映像創建一個單獨的部署，並將 CMD 覆蓋為：`python -m src.ingestion.cli ingest --auto-discover`。

---

## 3. GCP Cloud Run (雲端原生參考)

本節作為在 Google Cloud Platform 上部署高可用性、無伺服器 RAG 架構的技術展示。

### 架構藍圖
- **Web API**: 部署為 `Cloud Run Service` (自動擴展, 無狀態)。
- **ETL 爬蟲任務**: 部署為 `Cloud Run Job` (防止在長時間爬取任務中發生 CPU 限制/Throttling)。
- **金鑰管理**: 與 `GCP Secret Manager` 整合。

### 關鍵命令 (僅供參考)
```bash
# 部署 API 服務
./scripts/deploy.sh -e production -p your-project-id -r europe-west1

# 手動觸發作業
gcloud run jobs execute german-visa-rag-job-prod
```
*如需深入了解 GCP 部署腳本，請參閱 `infra/` 或 `scripts/` 目錄。*

---

## 4. 生產環境檢查清單

- [ ] **金鑰安全**: 沒有硬編碼的 API 金鑰；所有金鑰均通過環境變數或 Secret Manager 注入。
- [ ] **向量資料庫連接**: 已驗證與 Qdrant Cloud 的連接。
- [ ] **語義快取**: Redis 實例可連通（通過 `DEBUG` 日誌確認）。
- [ ] **引用來源**: 確保前端顯示 API 返回的 `metadata` 中的來源連結。
- [ ] **頻率限制**: 爬蟲已配置為對官方政府網站保持禮貌。

---

## 5. 故障排除與支援

### 常見問題
1. **OOM (內存不足)**: 如果運行重排器 (Reranker) 或查詢轉換器 (Query Transformer)，請確保容器至少有 2GB 內存。
2. **Qdrant 連接超時**: 檢查 Qdrant Cloud 白名單是否允許您的部署 IP（或設置 VPC 對等連接）。
3. **無效的 API 金鑰**: 驗證 `X-API-Key` 標頭是否與 `API_KEY` 環境變數匹配。

### 支援
如遇技術問題，請檢查 [GitHub Issues](https://github.com/yourusername/german-visa-rag/issues) 或諮詢系統日誌。