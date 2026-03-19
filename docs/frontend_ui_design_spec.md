# VisaPilot AI - 前端架構與設計書 (v1.0)

基於提供的渲染圖與現有 RAG 後端架構，本設計書定義了第一版前端應用程式的實作藍圖。

## 1. 核心定位與技術選型

*   **專案名稱**：VisaPilot AI
*   **定位**：展示德國簽證 RAG 系統強大檢索與分析能力的「工業級」火力展示平台。
*   **前端框架**：Vite + React 18 (或 Next.js 14 App Router, 若需 Server-side Rendering)
*   **樣式與 UI**：
    *   **Tailwind CSS**：用於快速構建佈局與實作 Glassmorphism。
    *   **Framer Motion**：用於對話氣泡彈出、側邊欄切換等微動畫。
    *   **Lucide React**：用於圖標（導覽列、來源圖示、發送按鈕等）。
*   **狀態管理**：Zustand (輕量級，適合處理跨面板的 UI 狀態同步)。
*   **API 請求**：Fetch API (原生支援 Streaming / SSE 處理對話)。

## 2. 視覺風格 (Visual Identity)

設計採用**現代深色模式 (Modern Dark Mode)** 與**玻璃擬態 (Glassmorphism)**：

*   **整體佈局 (Layout)**：三欄式整合儀表板 (Three-column Dashboard Bento Box)。
*   **背景 (Background)**：深邃的藏青色/石墨黑漸層 (例如 `#0B1120` 到 `#111827`)。
*   **面板材質 (Glass Panels)**：
    *   半透明背景 (`bg-white/5` 或 `bg-slate-800/40`)。
    *   背景模糊 (`backdrop-blur-md` 或 `backdrop-blur-xl`)。
    *   細緻的邊框 (`border border-white/10`)，強化前後景景深。
*   **強調色 (Accent Colors)**：
    *   主要按鈕與活動狀態：科技感十足的青藍色 (Cyan/Teal, 例如 `#06B6D4`)。
    *   文字顏色：高對比的白色 (`text-slate-100`) 與輔助的灰色 (`text-slate-400`)。

## 3. 模組與元件規劃 (Component Architecture)

畫面由三個主要直欄構成，完美對應使用者的視覺動線與 RAG 功能：

### 欄位一：左側導航與過濾列 (Left Sidebar - Control Panel)
負責全局導航與檢索上下文控制。

*   **App Logo**：VisaPilot AI (帶有科技感學士帽/簽證圖示)。
*   **Main Nav**：Home, My Profile, Documents, Settings。
*   **Recent Sources (信源過濾)**：
    *   展示權威來源圖示 (BMI, Make it in Germany, BAMF, AA)。
    *   *後端對接*：可作為全域的 `authority_level` 或 `source_url` 過濾器。
*   **Visa Categories (簽證類別選擇器)**：
    *   列出 `Chancenkarte`, `Skilled Worker Visa`, `Blue Card`, `Study Visa`。
    *   *後端對接*：點擊時，更新傳給後端的 `visa_types` enum 參數，讓 RAG 的 Hybrid Search 進行預先過濾 (Pre-filtering)。

### 欄位二：中央主對話區 (Center Area - Main Chat Interface)
處理核心互動與 RAG 內容展示。

*   **歷史對話區**：
    *   使用者對話氣泡與 AI 對話氣泡交互排列。
    *   **Streaming 支援**：AI 回應必須支援打字機流式輸出。
    *   **Source Chips (來源標籤元件)**：在 AI 回答中穿插或置底顯示資料來源 (例如 `Source: BMI.de`)。可點擊展開查看具體擷取的文本 (Chunk text) 與權威度分數。
*   **輸入框 (Message Input)**：
    *   懸浮於底部的玻璃面板輸入框。
    *   包含附件圖示 (未來支援上傳履歷分析)、語音輸入圖示 (麥克風) 與發送按鈕。

### 欄位三：右側動態分析面版 (Right Sidebar - Contextual Insights)
**這是將 RAG 昇華為 AI 助理的殺手級功能。** 將非結構化文字轉化為結構化資訊。

*   **動態進度追蹤 (Progress Checklist)**：
    *   根據當前對話上下文 (如 Chancenkarte)，自動高亮使用者目前處於哪個階段 (資格檢查 -> 分數計算 -> 文件清單)。
*   **核心條件摘要 (Requirement Summary)**：
    *   當 AI 生成針對特定簽證的回覆時，同步提取關鍵實體 (Entities)：例如「需要分數：6+ 分」、「主要條件：B1 德語、工作經驗、年齡 < 35」。
    *   *後端配合*：後端 `AnswerGenerator` 可以考慮在回傳文字的同時，利用 LLM 工具呼叫 (Function Calling) 額外回傳一個 JSON 結構，前端解析 JSON 後渲染於此面板。

## 4. API 對接計畫 (Backend Integration Points)

| 前端功能 | 後端 Endpoint (來自 `routes.py`) | 實作重點 |
| :--- | :--- | :--- |
| **發送訊息 (串流)** | `POST /query/ask/stream` | 解析 SSE (Server-Sent Events) 並將文字逐字渲染到畫面上。同時收集最後回傳的 `sources` 陣列。 |
| **渲染來源標籤** | SSE 結束後的 `sources` | 讀取 `metadata.source_title` 及 `metadata.authority_level` 來渲染帶有官方認證 Icon 的 Chip。 |
| **簽證類別過濾** | (需擴充 `QueryRequest`) | 目前 API 尚未接收 visa filter，建議在 `QueryRequest` 中加入 `visa_types` 欄位並傳遞給 `HybridRetriever`。 |
| **右側摘要面板** | (需擴充 `QueryResponse`) | 建議由 `answer_generator.py` 同步生成一份針對當前對話的 JSON 摘要，隨同回答一起回傳。 |

## 5. 開發階段建議 (Phases)

*   **Phase 1: 靜態 UI 雛形**：使用 Tailwind CSS 刻出三欄玻璃擬態版面，填入假資料，確認視覺感受。
*   **Phase 2: RAG API 串接**：實作中央對話區，對接 `/query/ask/stream`，確保流式打字與來源 (Source Chips) 能正確顯示。
*   **Phase 3: 狀態連動與過濾**：實作左側簽證分類點擊連動，並將參數帶入 API。
*   **Phase 4: 右側智慧面板**：後端擴充結構化輸出能力，前端實作動態 Checklist 與 Summary。
