"""LLM Factory for seamless fallback routing between OpenAI and local models."""

from typing import Optional
from src.config import settings
from src.logger import logger
from src.llm.openai_client import OpenAIClient
from src.llm.ollama_client import OllamaClient


class LLMFactory:
    """Factory to manage LLM instances and fallback strategies."""
    
    _instance = None

    @classmethod
    def get_client(cls):
        if cls._instance is not None:
            return cls._instance

        # 優先使用 OpenAI (檢查是否有設定真實的 API Key)
        if settings.openai_api_key and settings.openai_api_key != "sk-...your-key-here...":
            logger.info(f"Initializing primary LLM: OpenAI ({settings.openai_model})")
            cls._instance = OpenAIClient(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_api_base
            )
            
        # 雲端 Key 沒設定，且開啟了本地選項，則退避到 Ollama
        elif settings.use_ollama:
            logger.warning("OpenAI API Key not found or valid. Falling back to local Ollama!")
            cls._instance = OllamaClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model
            )
            
        else:
            # 如果都沒設定，還是預設給 OpenAI，讓它在實際呼叫時報錯
            logger.warning("No valid LLM configuration found. Defaulting to OpenAI (will likely fail on call).")
            cls._instance = OpenAIClient(
                api_key=settings.openai_api_key,
                model=settings.openai_model
            )
            
        return cls._instance


def get_llm_client():
    """Global dependency injection point for LLM client."""
    return LLMFactory.get_client()
