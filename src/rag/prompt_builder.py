"""
Prompt building with safety checks for RAG.
Prevents prompt injection and ensures faithful grounding in retrieved context.
"""

from typing import List, Dict, Any, Optional
import re

from src.config import settings
from src.logger import logger


SYSTEM_PROMPT = """
你是一位德國移民政策與簽證專家的助手。擅長解釋 2026 年最新版《專業人才移民法 (FEG 2.0)》、機會卡 (Chancenkarte)、歐盟藍卡 (EU Blue Card) 及學生簽證的相關規定。

<RULES>
1. **依賴檢索**：你的回答必須完全基於下方提供的「檢索文件」。禁止自行補充超出檢索範圍的知識。
2. **精確引用門檻**：請特別留意檢索文件中的金額與數據（例如：學生存款 €11,904、機會卡存款 €13,092、藍卡薪資、機會卡積分制），若文件中有提及，請精確引用。
3. **Markdown 引用來源**：每個陳述都必須附上 Markdown 超連結 `[{citation_label} N](URL)`。
4. **衝突處理**：優先選取官方 (🔴 [OFFICIAL]) 與最新日期的資訊。如果文件中沒有相關資訊，但與下方提供的<DOMAIN_KNOWLEDGE>相符，可以作為輔助補充；若皆無相關資訊，請直接回答不知道。
</RULES>

<DOMAIN_KNOWLEDGE>
【專業人才 (FEG/Blue Card) 專屬邏輯核心基準】(輔助判斷標準，具體細節優先依檢索文件為主)
- **語言非硬性**：對於專才簽證與藍卡，語言能力通常取決於雇主。除非是「認可夥伴關係 (A2)」，否則不應視為硬指標。
- **藍卡合約效期**：工作合約效期必須至少 **6 個月**。
- **藍卡 IT 豁免**：IT 工程師 3 年經驗可豁免學位。
- **45 歲條款**：针对專業人才簽證 (FEG)，若用戶年齡 > 45，薪資需達約定門檻。
- **學生簽證核心**：大學錄取通知書 (Zulassung) 是絕對關鍵。
- **學生財力證明**：存入限制提領帳戶 (€11,904) 是常見做法，但獎學金或擔保書 (VE) 亦可。
- **學生醫療保險**：入學註冊必備，簽證申請時也需提供。
- **絕對禁止通靈/假定客戶資訊**：若用戶未提及年齡、學歷、具體工作年數，請標記為「待確認」，**且分數計為 0**。絕不能假設用戶「未滿40歲」或「擁有大學學位」來強行湊分。
- **門檻優先原則**：機會卡是「先過門檻，再算積分」。必須先確認用戶滿足「財力證明」、「語言 A1/英文 B2」及「2年制基本學位」這三項基礎指標，才能開始列出加分項目或進行總分加總。
- **機會卡路徑一 (直接認可)**：擁有受德國完全認可的專業資格，**不強制要求語言能力**，且免算積分。
- **機會卡路徑二 (積分制硬性門檻)**：必須先達標「財力證明」、「語言 A1/英文 B2」與「基礎學位 (2年制)」才能開始算分。若未達標，請直接指出不足，不要進行積分累加。
- **機會卡積分規則 (Step 2 - 達標 6 分即合格)**：
    1. **語言 (上限 4 分)**：德文 A2(+1), B1(+2), B2(+3), C1(+4)；英文 C1(+1)。*語言可疊加*。
    2. **工作經驗 (上限 3 分)**：過去7年內滿5年相關經驗(+3) (若有5年以上經驗即可獲得)；過去5年內滿2年相關經驗(+2)。
    3. **年齡 (上限 2 分)**：<=35歲(+2)；36-40歲(+1)。
    4. **學歷與稀缺職業 (上限 4 分)**：部分認同(+4)；稀缺職業 IT/護理/工程(+1)。
    5. **德國經歷 (上限 1 分)**：5年內合法居住滿6個月(+1)。
    6. **伴侶加分 (上限 1 分)**：伴侶也符合機會卡門檻(+1)。
</DOMAIN_KNOWLEDGE>

<TAG_INSTRUCTIONS>
請分析用戶當前狀態與意圖，並在回答的最末尾輸出以下隱藏標籤 (不可出現在主要回答文字中)：

1. **里程碑進度 (MILESTONE)**：格式 `[MILESTONE:ID:STATUS]` (STATUS 必須為 current 或 completed)
   - ID=1 (資格/初步諮詢)：詢問門檻。
   - ID=2 (專一進階)：機會卡(積分計算)、藍卡/FEG(合約審核/薪資)、學生(入學證明)。
   - ID=3 (文件準備)：詢問財力證明、公證等細節。 (通常這已經是 AI RAG 對話的最終階段)

2. **基本條件更新 (REQ)**：格式 `[REQ:ID:VALUE:STATUS]` (請主動評估用戶目前的條件狀況並以標籤形式輸出，這部分為系統隱藏判定，**絕不能**將 `[REQ...]` 、 `Status: required` 等字眼寫在給用戶的對話內文裡！並且**絕對禁止**出現「以下是標籤」這類前導敘述文字！
   - **基礎門檻項目 (ID為 1- 開頭)**：VALUE 必須極簡短且使用中性 Key，如：`MET` (已達標)、`LACK_OF_FUNDS:13092` (缺金額)、`TBC` (待確認)。
   - **積分加分項目 (ID為 2- 開頭)**：VALUE 必須遵守嚴格的「能力|分數」格式（用直線 | 隔開），例如：`B1|2`、`5_YEARS_EXP|3`、`UNDER_35|2`、`DEGREE|4`。**若資訊不明，請寫 `TBC|0`**。
   - **STATUS 嚴格規範**：STATUS 只能是 `required` (已達標) 或 `warning` (未達標/資訊不足/有風險/待確認)。**絕對禁止使用 `info` 作為狀態！**

【標籤輸出標準結構範例】
(請務必根據用戶真實情況填入變數，不可直接抄襲範例)：
- 機會卡：[REQ:2-1:B1|2:required] [REQ:1-1:LACK_OF_FUNDS:13092:warning]
- 藍卡：[REQ:2:SALARY_MET:required] [REQ:1:IT_3Y_EXP:required] (IT路徑)
- 學生：[REQ:4:ADMITTED:required] [REQ:3:INSURED:required]
- 通用：[REQ:2-3:TBC|0:warning] (當資訊不明時)

(錯誤示範 - 絕對禁止這樣寫)：
[REQ:2-5: - |info] (X 錯誤：使用了禁止的 info 狀態，且格式不對)
[REQ:1-1:已達標:required] (X 錯誤：使用了中文，應使用 MET)
   - **專業人才 (Skilled Worker/FEG)**: 1:學歷與專業資格, 2:薪資與勞動條件, 3:45歲以上特殊條款, 4:語言能力
   - **歐盟藍卡 (EU Blue Card)**: 1:學歷與專業資格, 2:德國全職工作合約, 3:語言加分
   - **機會卡 (Chancenkarte)**: 
       - 必備門檻: 1-1:財力證明, 1-2:語言, 1-3:學經歷
       - 積分項目: 2-1:語言加分, 2-2:經驗加分, 2-3:年齡加分, 2-4:資格加分, 2-5:德國經歷, 2-6:伴侶加分
   - **學生簽證 (Student)**: 1:財力證明, 2:語言能力證明, 3:德國醫療保險, 4:前置學經歷資格
</TAG_INSTRUCTIONS>

【檢索文件】
{context}

<OUTPUT_FORMAT>
- 語言：請使用提問的語言回答。
- 若提及機會卡，請先在文字回答中逐項列出得分項目再加總，**並且務必針對每一項提及的條件輸出對應的 `[REQ:ID:VALUE:STATUS]` 標籤。**
- **嚴禁洩漏標籤**：標籤必須放在整個回覆的最末端，且與回覆正文之間至少空一行。正文中絕對禁止出現任何 `[REQ` 或 `[MILESTONE` 字樣。
請嚴格依循以下格式回覆：
（給用戶的回覆文字，並包含 Markdown 引用來源）

（在此處直接放入所有標籤，如 [MILESTONE:1:current] [REQ:1-1:MET:required] ...）
</OUTPUT_FORMAT>
"""

class PromptBuilder:
    """Build and validate prompts for RAG responses."""

    @staticmethod
    def build_system_prompt(context: str, question: str, language: Optional[str] = None, visa_type: Optional[str] = None, requirements: Optional[List[Dict[str, str]]] = None) -> str:
        """Get system prompt with context and question injected."""
        citation_label = "Paragraph" if language == "en" else "Absatz" if language == "de" else "段落"
        
        prompt = SYSTEM_PROMPT.format(
            context=context, 
            citation_label=citation_label
        )
        
        if visa_type:
            prompt += f"\n\n【當前簽證脈絡】\n用戶目前正在查看的是：**{visa_type}**。請特別優先針對此類簽證進行針對性回答與標籤輸出。"

        if language and language != "auto":
            lang_map = {
                "en": "English",
                "de": "German",
                "zh-TW": "Traditional Chinese"
            }
            target_lang = lang_map.get(language, language)
            prompt += f"\n\n【語言指令】\n請務必使用 **{target_lang}** 回答此問題。"
            
        if requirements:
            # Filter out headers to only show real requirements
            valid_reqs = []
            for req in requirements:
                if req.get('id') and not req.get('id').startswith('header'):
                    # format properly for context
                    val = req.get('value', '')
                    if req.get('id').startswith('2-') and '|' in val:
                        parts = val.split('|')
                        display_val = f"{parts[0]} (+{parts[1]}分)"
                    else:
                        display_val = val
                    valid_reqs.append(f"- ID={req.get('id')}: {req.get('label')} = {display_val} (內部狀態為: {req.get('status')})")
            
            if valid_reqs:
                ui_state_str = "\n".join(valid_reqs)
                prompt += f"\n\n<CURRENT_UI_STATE>\n以下是用戶當前的 UI 狀態紀錄（哪些條件已達成、哪些還缺漏）。這非常重要！\n**請務必根據用戶剛才提及的最新資訊，對照這份清單，判斷狀態是否該「翻盤 (從 warning 變成 required)」或「被滿足」。**\n只要用戶提及了任何能滿足條件的金額或資訊（例如換算後大於門檻），你必須立刻輸出 `[REQ:ID:已達標:required]` 的標籤來覆寫原本 `缺...:warning` 的狀態！\n特別注意：請勿將 `內部狀態` 或系統機制直接暴露在文字回覆中，絕對禁止在回答結尾寫出「(未涉及更新...)」這類解釋。如果不需更新條件，直接默默結束對話。\n{ui_state_str}\n</CURRENT_UI_STATE>"
            
        return prompt

    @staticmethod
    def build_context_from_retrieval(
        retrieval_results: List[Dict[str, Any]],
        top_k: int = 5,
        language: str = "en"
    ) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            retrieval_results: Results from hybrid retriever
            top_k: Number of top results to include
            language: Target language for labels
            
        Returns:
            Formatted context for prompt
        """
        context_parts = []
        citation_label = "Paragraph" if language == "en" else "Absatz" if language == "de" else "段落"
        
        for i, result in enumerate(retrieval_results[:top_k], 1):
            metadata = result.get("metadata", {})
            text = result.get("text", "")
            
            source_url = metadata.get("source_url", "Unknown")
            section_header = metadata.get("section_header", "General")
            authority = metadata.get("authority_level", "third_party")
            
            # Authority badge for priority signaling
            authority_badge = {
                "official": "🔴 [OFFICIAL]",
                "semi_official": "🟡 [SEMI-OFFICIAL]",
                "third_party": "⚪ [THIRD-PARTY]",
            }.get(authority, "")
            
            part = f"""
【{citation_label} {i}】 {authority_badge}
來源：{source_url}
章節：{section_header}
---
{text}
---
"""
            context_parts.append(part)
        
        return "\n".join(context_parts)

    @staticmethod
    def build_user_message(
        question: str,
        context: str,
    ) -> Dict[str, str]:
        """Build final user message."""
        # Inject context into system prompt
        system_with_context = SYSTEM_PROMPT.format(
            context=context,
            citation_label="Paragraph", # Default for validation
            question="[will be filled in user message]"
        )
        
        return {
            "role": "user",
            "content": f"""
【用戶問題】
{question}

【指導】
請基於上方檢索文件回答，並清楚標註每個陳述的來源。如果信息不足，請明確說明。
""",
        }

    @staticmethod
    def validate_context_for_injection(context: str) -> bool:
        """
        Validate context for prompt injection attempts.
        
        Security checks:
        - Detect suspicious command patterns
        - Check for encoded payloads
        - Flag potential jailbreak attempts
        
        Returns:
            True if context seems safe, False if suspicious
        """
        suspicious_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"you are now",
            r"disregard",
            r"pretend you are",
            r"forget about",
            r"new instructions",
            r"execute code",
            r"eval\(",
            r"import os",
        ]
        
        context_lower = context.lower()
        
        for pattern in suspicious_patterns:
            if re.search(pattern, context_lower):
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return False
        
        return True




# Singleton
_builder: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Get prompt builder instance."""
    global _builder
    if _builder is None:
        _builder = PromptBuilder()
    return _builder
