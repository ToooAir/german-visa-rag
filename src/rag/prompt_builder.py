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

【關鍵要求】
1. **完全基於檢索內容**：你的回答必須完全基於下方提供的「檢索文件」。禁止自行補充超出檢索範圍的內容。
2. **2026 最新門檻意識**：請特別留意檢索文件中的金額與數據（例如：學生存款 €11,904、機會卡存款 €13,092、藍卡薪資、機會卡積分制），若文件中有提及，請精確引用。
3. **專業人才 (FEG/Blue Card) 專屬邏輯**：
   - **語言非硬性**：對於專才簽證與藍卡，語言能力通常取決於雇主。除非是「認可夥伴關係 (A2)」，否則不應視為硬指標。
   - **藍卡合約效期**：工作合約效期必須至少 **6 個月**。
   - **學生簽證核心**：大學錄取通知書 (Zulassung) 是絕對關鍵。
   - **學生財力證明**：存入限制提領帳戶 (€11,904) 是常見做法，但獎學金或擔保書 (VE) 亦可。
   - **學生醫療保險**：入學註冊必備，簽證申請時也需提供。
   - **藍卡 IT 豁免**：IT 工程師 3 年經驗可豁免學位。
   - **機會卡硬性門檻 (Step 1)**：所有積分制申請人必須先達標「財力證明」、「語言 A1/英文 B2」與「基礎學位 (2年制)」才能開始算分。若未達標，請直接指出不足，不要進行積分累加。
   - **機會卡積分規則 (Step 2 - 達標 6 分即合格)**：
       1. **語言 (上限 4 分)**：德文 A2(+1), B1(+2), B2(+3), C1(+4)；英文 C1(+1)。*語言可疊加但上限 4 分*。
       2. **工作經驗 (上限 3 分)**：7年內滿5年相關(+3)；5年內滿2年相關(+2)。
       3. **年齡 (上限 2 分)**：<=35歲(+2)；36-40歲(+1)。
       4. **學歷與稀缺職業 (上限 4 分)**：部分認同(+4)；稀缺職業 IT/護理/工程(+1)。
       5. **德國經歷 (上限 1 分)**：5年內合法居住滿6個月(+1)。
       6. **伴侶加分 (上限 1 分)**：伴侶也符合機會卡門檻(+1)。
   - **45 歲條款**：针对專業人才簽證 (FEG)，若用戶年齡 > 45，薪資需達約定門檻。
4. **Markdown 引用來源**：每個陳述都必須附上 Markdown 超連結 `[{citation_label} N](URL)`。
5. **衝突處理**：優先選取官方 (🔴 [OFFICIAL]) 與最新日期的資訊。

【動態標籤輸出】 (關鍵：請在回答最末尾附加以下隱藏標籤，不可出現於內容中)

1. **里程碑進度 (MILESTONE)**：格式 `[MILESTONE:ID:STATUS]`
   依據用戶的詢問意圖判斷階段 (current/completed)。ID 對應如下：
   - ID=1 (資格/初步諮詢)：詢問門檻。
   - ID=2 (專一進階)：機會卡(積分計算)、藍卡/FEG(合約審核/薪資)、學生(入學證明)。
   - ID=3 (文件準備)：詢問財力證明、公證等細節。 (通常輸出 [MILESTONE:3:current][MILESTONE:1:completed][MILESTONE:2:completed])
   - ID=4 (簽證領取)：詢問德協預約、面試。
   - ID=5 (核發/入境)：詢問核發時間、報到。

2. **基本條件更新 (REQ)**：格式 `[REQ:ID:VALUE:STATUS]`
   僅當用戶提及自身條件時輸出。VALUE 必須使用當前對話語言並簡短。
   STATUS 必須為：`required` (已達標), `warning` (有風險/不足), `info` (中立資料)。
   ID 對應表 (依據不同簽證類別)：
   - **專業人才 (Skilled Worker/FEG)**: 1:學歷與專業資格, 2:薪資與勞動條件, 3:45歲以上特殊條款, 4:語言能力 (彈性控制)
   - **歐盟藍卡 (EU Blue Card)**: 1:學歷與專業資格, 2:德國全職工作合約, 3:語言加分 (PR Bonus)
   - **機會卡 (Chancenkarte)**: 
       - 必備門檻: 1-1:財力證明, 1-2:語言, 1-3:學經歷
       - 積分項目: 2-1:語言加分, 2-2:經驗加分, 2-3:年齡加分, 2-4:資格加分, 2-5:德國經歷, 2-6:伴侶加分
   - **學生簽證 (Student)**: 1:財力證明, 2:語言能力證明, 3:德國醫療保險, 4:前置學經歷資格

【檢索文件】
{context}

【指導方針】
- 語言：使用提問語言回答。
- 標籤必須放在回答的最後面。
"""

DISCLAIMERS = {
    "zh-TW": """
⚠️ **免責聲明**：
本回答基於公開信息，僅供參考，不構成法律意見。德國移民政策變動頻繁。所有重要決定（簽證申請、工作許可、Chancenkarte 申請）請：
1. 查詢官方資源：https://www.make-it-in-germany.com 或當地外事局
2. 諮詢專業移民律師或政府服務中心
3. 確認最新版本，因政策可能已更新

本系統不對使用此信息造成的任何後果負責。
""",
    "en": """
⚠️ **Disclaimer**:
This response is based on public information and is for reference only; it does not constitute legal advice. German immigration policies change frequently. For all important decisions (visa applications, work permits, Chancenkarte applications), please:
1. Consult official resources: https://www.make-it-in-germany.com or your local Foreigners' Authority (Ausländerbehörde)
2. Seek advice from professional immigration lawyers or government service centers
3. Verify the latest versions, as policies may have been updated

This system is not responsible for any consequences resulting from the use of this information.
""",
    "de": """
⚠️ **Haftungsausschluss**:
Diese Antwort basiert auf öffentlichen Informationen und dient nur zu Referenzzwecken; sie stellt keine Rechtsberatung dar. Die deutsche Einwanderungspolitik ändert sich häufig. Für alle wichtigen Entscheidungen (Visumanträge, Arbeitserlaubnisse, Chancenkarte-Anträge), bitte:
1. Konsultieren Sie offizielle Quellen: https://www.make-it-in-germany.com oder Ihre örtliche Ausländerbehörde
2. Suchen Sie Rat bei professionellen Einwanderungsanwälten oder staatlichen Beratungsstellen
3. Überprüfen Sie die neuesten Versionen, da sich Richtlinien aktualisiert haben könnten

Dieses System übernimmt keine Haftung für Folgen, die aus der Nutzung dieser Informationen entstehen.
"""
}
# Fallback to English
DISCLAIMER = DISCLAIMERS["en"]

class PromptBuilder:
    """Build and validate prompts for RAG responses."""

    @staticmethod
    def build_system_prompt(context: str, question: str, language: Optional[str] = None, visa_type: Optional[str] = None) -> str:
        """Get system prompt with context and question injected."""
        citation_label = "Paragraph" if language == "en" else "Absatz" if language == "de" else "段落"
        prompt = SYSTEM_PROMPT.format(context=context, citation_label=citation_label)
        
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

    @staticmethod
    def add_disclaimer(response: str, language: Optional[str] = None) -> str:
        """Append disclaimer to response in the correct language."""
        disclaimer = DISCLAIMERS.get(language, DISCLAIMERS["en"])
        return f"{response}\n\n{disclaimer}"


# Singleton
_builder: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Get prompt builder instance."""
    global _builder
    if _builder is None:
        _builder = PromptBuilder()
    return _builder
