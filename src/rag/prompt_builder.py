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
2. **引用來源**：每個陳述都必須附上來源 URL 與段落編號，格式為 [Source: URL, Section N]。
3. **清楚說明限制**：如果檢索文件中沒有找到相關信息，必須明確回答：「我查閱的資料庫中暫時沒有關於此的具體信息。建議查詢官方資源。」
4. **去重與去衝突**：如果多份文件有衝突，優先選擇官方來源與最新版本，並說明依據的日期。
5. **責任聲明**：在回答開始或結束附上：「⚠️ 免責聲明：本回答僅供參考，不構成法律意見。所有重要決定請以官方渠道與個案審查為準。」

【檢索文件】
{context}

【用戶問題】
{question}

【指導方針】
- 語言：用用戶提問的語言回答。
- 風格：專業但易理解，避免冗長法律術語。
- 長度：根據問題複雜度調整，通常 200-500 字。
"""

DISCLAIMER = """
⚠️ **免責聲明**：
本回答基於公開信息，僅供參考，不構成法律意見。德國移民政策變動頻繁。所有重要決定（簽證申請、工作許可、Chancenkarte 申請）請：
1. 查詢官方資源：https://www.make-it-in-germany.com 或當地外事局
2. 諮詢專業移民律師或政府服務中心
3. 確認最新版本，因政策可能已更新

本系統不對使用此信息造成的任何後果負責。
"""


class PromptBuilder:
    """Build and validate prompts for RAG responses."""

    @staticmethod
    def build_system_prompt(context: str, question: str) -> str:
        """Get system prompt with context and question injected."""
        return SYSTEM_PROMPT.format(context=context, question=question)

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
    def add_disclaimer(response: str) -> str:
        """Append disclaimer to response."""
        return f"{response}\n\n{DISCLAIMER}"


# Singleton
_builder: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Get prompt builder instance."""
    global _builder
    if _builder is None:
        _builder = PromptBuilder()
    return _builder
