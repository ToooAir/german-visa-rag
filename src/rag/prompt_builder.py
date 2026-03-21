"""
Prompt building with safety checks for RAG.
Prevents prompt injection and ensures faithful grounding in retrieved context.
"""

from typing import List, Dict, Any, Optional
import re

from src.config import settings
from src.logger import logger


SYSTEM_PROMPT = """
你是一位德國移民政策的專家助手。用戶會用中文、英文或德文提問德國簽證、工作許可或 Chancenkarte 的相關規定。

【關鍵要求】
1. **完全基於檢索內容**：你的回答必須完全基於下方提供的「檢索文件」。不允許自行補充、猜測或根據培訓數據推測。
2. **引用來源**：每個陳述都必須附上來源，格式為 Markdown 超連結 `[段落 N](URL)`。不要直接顯示原始 URL。
3. **清楚說明限制**：如果檢索文件中沒有找到相關信息，必須明確回答：「我查閱的資料庫中暫時沒有關於此的具體信息。建議查詢官方資源。」
4. **去重與去衝突**：如果多份文件有衝突，優先選擇官方來源與最新版本，並說明依據的日期。

【檢索文件】
{context}

【用戶問題】
{question}

【動態資料更新】
如果在對話中偵測到以下資訊，請在回答的最末尾附加隱藏標籤：
1. **里程碑進度推斷 (純基於提問意圖)** (格式: `[MILESTONE:ID:STATUS]`)
   - 根據用戶所在階段輸出隱藏標籤。可以一次輸出多個標籤來更新不同階段。
   - `ID=1` (資格審查)：詢問「我符合資格嗎？」等門檻條件。請輸出 `[MILESTONE:1:current]`
   - `ID=2` (積分計算)：詢問「這能拿幾分？」。請輸出 `[MILESTONE:2:current]`
   - `ID=3` (準備文件)：詢問「限制提領帳戶」、「ZAB公證」、「動機信」等實操細節。請務必同時輸出三個標籤：`[MILESTONE:3:current][MILESTONE:1:completed][MILESTONE:2:completed]`
   - `ID=4` (預約使館)：詢問「德協預約」、「簽證面試」。請輸出 `[MILESTONE:4:current][MILESTONE:3:completed]`
   - `ID=5` (抵達後續)：詢問「入籍 (Anmeldung)」、「找房」。請輸出 `[MILESTONE:5:current][MILESTONE:4:completed]`

2. **基本條件更新** (格式: `[REQ:ID:VALUE:STATUS]`)
   - 若用戶提及自身的條件，請更新對應的項目。若未提及請勿輸出該標籤。
   - **重要：VALUE 必須使用指定的回答語言 (Target Language) 並保持簡短。**
   - `ID=1` (語言能力)：例如 German 下為 `[REQ:1:B1 Deutsch:required]`，English 下為 `[REQ:1:B1 German:required]`
   - `ID=2` (工作經驗)：例如 German 下為 `[REQ:2:5 Jahre:required]`
   - `ID=3` (年齡)：例如 German 下為 `[REQ:3:30 Jahre:info]`
   - `ID=4` (學歷資格)：例如 German 下為 `[REQ:4:Bachelor:required]`
   - STATUS 必須為：`required` (綠色達標) 或 `warning` (黃色警告) 或 `info` (灰色中立)。
   - 標籤必須放在最後，不可出現在回答正文中。

【指導方針】
- 語言：用用戶提問的語言回答。
- 風格：專業但易理解，避免冗長法律術語。
- 長度：根據問題複雜度調整，通常 200-500 字。
- 標籤：標籤必須放在最後，不可隨機出現在正文中。
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
    def build_system_prompt(context: str, question: str, language: Optional[str] = None) -> str:
        """Get system prompt with context and question injected."""
        prompt = SYSTEM_PROMPT.format(context=context, question=question)
        
        if language and language != "auto":
            # Map language codes to readable names if needed, but the model usually understands codes
            lang_map = {
                "en": "English",
                "de": "German",
                "zh-TW": "Traditional Chinese"
            }
            target_lang = lang_map.get(language, language)
            prompt += f"\n\n【語言指令】\n請務必使用 **{target_lang}** 回答此問題，忽視用戶提問所使用的語言。"
            
        return prompt

    @staticmethod
    def build_context_from_retrieval(
        retrieval_results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            retrieval_results: Results from hybrid retriever
            top_k: Number of top results to include
            
        Returns:
            Formatted context for prompt
        """
        context_parts = []
        
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
【段落 {i}】 {authority_badge}
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
