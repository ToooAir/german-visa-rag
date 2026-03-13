"""
Configuration management with environment variable support.
Centralized settings for LLM, Vector DB, Cache, and Observability.
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""

    # ============================================
    # Environment & Debug
    # ============================================
    environment: Literal["development", "staging", "production"] = Field(
        default="development", env="ENVIRONMENT"
    )
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # ============================================
    # OpenAI LLM Configuration
    # ============================================
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1", env="OPENAI_API_BASE"
    )
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    
    # Token limits & cost control
    max_query_tokens: int = Field(default=256, env="MAX_QUERY_TOKENS")
    max_context_tokens: int = Field(default=4000, env="MAX_CONTEXT_TOKENS")
    max_response_tokens: int = Field(default=1024, env="MAX_RESPONSE_TOKENS")

    # ============================================
    # Ollama (Local LLM Fallback)
    # ============================================
    use_ollama: bool = Field(default=False, env="USE_OLLAMA")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="mistral", env="OLLAMA_MODEL")

    # ============================================
    # Qdrant Vector Database
    # ============================================
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="german-visa-docs", env="QDRANT_COLLECTION_NAME"
    )
    qdrant_vector_size: int = Field(default=1536, env="QDRANT_VECTOR_SIZE")
    qdrant_prefer_grpc: bool = Field(default=False, env="QDRANT_PREFER_GRPC")

    # ============================================
    # Redis Cache
    # ============================================
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    enable_query_cache: bool = Field(default=True, env="ENABLE_QUERY_CACHE")

    # ============================================
    # SQLite State Store
    # ============================================
    sqlite_db_path: Path = Field(
        default=Path("./data/state.db"), env="SQLITE_DB_PATH"
    )

    # ============================================
    # API Security
    # ============================================
    api_key: str = Field(default="dev-key-12345", env="API_KEY")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")

    # ============================================
    # Ingestion Pipeline
    # ============================================
    ingestion_schedule_interval_hours: int = Field(
        default=24, env="INGESTION_SCHEDULE_INTERVAL_HOURS"
    )
    crawler_rate_limit_requests_per_second: float = Field(
        default=2.0, env="CRAWLER_RATE_LIMIT_REQUESTS_PER_SECOND"
    )
    crawler_timeout_seconds: int = Field(default=30, env="CRAWLER_TIMEOUT_SECONDS")
    crawler_max_retries: int = Field(default=3, env="CRAWLER_MAX_RETRIES")
    crawler_user_agent: str = Field(
        default="Mozilla/5.0 (German-Visa-RAG/1.0)",
        env="CRAWLER_USER_AGENT",
    )

    # ============================================
    # Chunking Configuration
    # ============================================
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")
    parent_chunk_size: int = Field(default=2048, env="PARENT_CHUNK_SIZE")

    # ============================================
    # Retrieval Configuration
    # ============================================
    retrieval_top_k_hybrid: int = Field(default=20, env="RETRIEVAL_TOP_K_HYBRID")
    retrieval_top_k_reranked: int = Field(default=10, env="RETRIEVAL_TOP_K_RERANKED")
    retrieval_dense_weight: float = Field(default=0.7, env="RETRIEVAL_DENSE_WEIGHT")
    retrieval_sparse_weight: float = Field(default=0.3, env="RETRIEVAL_SPARSE_WEIGHT")

    # ============================================
    # Reranker Configuration
    # ============================================
    reranker_api_type: Literal["mock", "cohere", "jina"] = Field(
        default="mock", env="RERANKER_API_TYPE"
    )
    reranker_api_key: Optional[str] = Field(default=None, env="RERANKER_API_KEY")
    reranker_model_name: str = Field(default="rerank-english-v2.0", env="RERANKER_MODEL_NAME")

    # ============================================
    # MLflow Tracking
    # ============================================
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="german-visa-rag", env="MLFLOW_EXPERIMENT_NAME")
    enable_mlflow: bool = Field(default=True, env="ENABLE_MLFLOW")

    # ============================================
    # Observability
    # ============================================
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    enable_prometheus: bool = Field(default=False, env="ENABLE_PROMETHEUS")

    # ============================================
    # Paths
    # ============================================
    seed_urls_path: Path = Field(default=Path("src/ingestion/seed_urls.yml"), env="SEED_URLS_PATH")
    logs_dir: Path = Field(default=Path("./logs"), env="LOGS_DIR")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator("sqlite_db_path", "logs_dir", pre=True)
    def ensure_path_exists(cls, v):
        """Ensure directories exist."""
        p = Path(v)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
