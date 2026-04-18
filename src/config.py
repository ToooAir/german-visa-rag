"""
Configuration management with environment variable support.
Centralized settings for LLM, Vector DB, Cache, and Observability.
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""

    # ============================================
    # Environment & Debug
    # ============================================
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development", validation_alias="ENVIRONMENT"
    )
    debug: bool = Field(default=False, validation_alias="DEBUG")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # ============================================
    # OpenAI LLM Configuration
    # ============================================
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    # Separate model for QueryTransformer (cheaper nano model; falls back to openai_model if unset)
    query_transform_model: str = Field(default="gpt-4.1-nano-2025-04-14", validation_alias="QUERY_TRANSFORM_MODEL")
    openai_api_base: str = Field(default="https://api.openai.com/v1", validation_alias="OPENAI_API_BASE")
    embedding_model: str = Field(default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1536, validation_alias="EMBEDDING_DIMENSION")

    # Azure OpenAI Configuration
    use_azure_openai: bool = Field(default=False, validation_alias="USE_AZURE_OPENAI")
    azure_openai_api_key: Optional[str] = Field(default=None, validation_alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, validation_alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2024-12-01-preview", validation_alias="AZURE_OPENAI_API_VERSION")
    azure_embedding_deployment: Optional[str] = Field(default=None, validation_alias="AZURE_EMBEDDING_DEPLOYMENT")
    azure_llm_deployment: Optional[str] = Field(default=None, validation_alias="AZURE_LLM_DEPLOYMENT")
    # Separate Azure deployment for QueryTransformer (e.g. a nano deployment); falls back to azure_llm_deployment
    azure_query_transform_deployment: Optional[str] = Field(
        default=None, validation_alias="AZURE_QUERY_TRANSFORM_DEPLOYMENT"
    )

    # Token limits & cost control
    max_query_tokens: int = Field(default=256, validation_alias="MAX_QUERY_TOKENS")
    max_context_tokens: int = Field(default=4000, validation_alias="MAX_CONTEXT_TOKENS")
    max_response_tokens: int = Field(default=1024, validation_alias="MAX_RESPONSE_TOKENS")
    api_timeout_seconds: int = Field(default=30, validation_alias="API_TIMEOUT_SECONDS")

    # Response / request body size guards (OOM protection)
    # ~50 000 chars ≈ 12 500 tokens — well above any expected answer length.
    max_response_chars: int = Field(default=50_000, validation_alias="MAX_RESPONSE_CHARS")
    # 2 000 chars is generous for a single query; rejects obviously abusive payloads.
    max_query_chars: int = Field(default=2_000, validation_alias="MAX_QUERY_CHARS")

    # ============================================
    # Ollama (Local LLM Fallback)
    # ============================================
    use_ollama: bool = Field(default=False, validation_alias="USE_OLLAMA")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="mistral", validation_alias="OLLAMA_MODEL")

    # ============================================
    # Qdrant Vector Database
    # ============================================
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="german-visa-docs", validation_alias="QDRANT_COLLECTION_NAME")
    qdrant_vector_size: int = Field(default=1536, validation_alias="QDRANT_VECTOR_SIZE")
    qdrant_prefer_grpc: bool = Field(default=False, validation_alias="QDRANT_PREFER_GRPC")

    # ============================================
    # Redis Cache
    # ============================================
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    cache_ttl_seconds: int = Field(default=3600, validation_alias="CACHE_TTL_SECONDS")
    enable_query_cache: bool = Field(default=True, validation_alias="ENABLE_QUERY_CACHE")

    # ============================================
    # SQLite State Store
    # ============================================
    sqlite_db_path: Path = Field(default=Path("./data/state.db"), validation_alias="SQLITE_DB_PATH")

    # ============================================
    # API Security
    # ============================================
    api_key: str = Field(..., validation_alias="API_KEY")
    api_key_header: str = Field(default="X-API-Key", validation_alias="API_KEY_HEADER")
    require_api_key: bool = Field(default=True, validation_alias="REQUIRE_API_KEY")
    allowed_hosts: Union[List[str], str] = Field(default=["localhost", "127.0.0.1"], validation_alias="ALLOWED_HOSTS")
    allowed_origins: Union[List[str], str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"], validation_alias="ALLOWED_ORIGINS"
    )

    # ============================================
    # Rate Limiting
    # ============================================
    enable_rate_limit: bool = Field(default=True, validation_alias="ENABLE_RATE_LIMIT")
    rate_limit_requests_per_minute: int = Field(default=20, validation_alias="RATE_LIMIT_REQUESTS_PER_MINUTE")

    # ============================================
    # Ingestion Pipeline
    # ============================================
    ingestion_schedule_interval_hours: int = Field(default=24, validation_alias="INGESTION_SCHEDULE_INTERVAL_HOURS")
    crawler_rate_limit_requests_per_second: float = Field(
        default=2.0, validation_alias="CRAWLER_RATE_LIMIT_REQUESTS_PER_SECOND"
    )
    crawler_timeout_seconds: int = Field(default=30, validation_alias="CRAWLER_TIMEOUT_SECONDS")
    crawler_max_retries: int = Field(default=3, validation_alias="CRAWLER_MAX_RETRIES")
    crawler_user_agent: str = Field(
        default="Mozilla/5.0 (German-Visa-RAG/1.0)",
        validation_alias="CRAWLER_USER_AGENT",
    )
    crawler_max_depth: int = Field(default=3, validation_alias="CRAWLER_MAX_DEPTH")
    crawler_max_pages_per_domain: int = Field(default=100, validation_alias="CRAWLER_MAX_PAGES_PER_DOMAIN")
    crawler_respect_robots_txt: bool = Field(default=True, validation_alias="CRAWLER_RESPECT_ROBOTS_TXT")
    crawler_discovery_enabled: bool = Field(default=True, validation_alias="CRAWLER_DISCOVERY_ENABLED")
    discovery_cache_ttl_hours: int = Field(default=168, validation_alias="DISCOVERY_CACHE_TTL_HOURS")

    # Internal APScheduler (Disable in cloud environments, use external Cron instead)
    enable_internal_scheduler: bool = Field(default=False, validation_alias="ENABLE_INTERNAL_SCHEDULER")

    # ============================================
    # Chunking Configuration
    # ============================================
    chunk_size: int = Field(default=512, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, validation_alias="CHUNK_OVERLAP")
    parent_chunk_size: int = Field(default=2048, validation_alias="PARENT_CHUNK_SIZE")

    # ============================================
    # Retrieval Configuration
    # ============================================
    retrieval_top_k_hybrid: int = Field(default=20, validation_alias="RETRIEVAL_TOP_K_HYBRID")
    retrieval_top_k_reranked: int = Field(default=10, validation_alias="RETRIEVAL_TOP_K_RERANKED")
    retrieval_dense_weight: float = Field(default=0.7, validation_alias="RETRIEVAL_DENSE_WEIGHT")
    retrieval_sparse_weight: float = Field(default=0.3, validation_alias="RETRIEVAL_SPARSE_WEIGHT")
    enable_sparse_search: bool = Field(default=True, validation_alias="ENABLE_SPARSE_SEARCH")
    sparse_vocab_size: int = Field(default=30_000, validation_alias="SPARSE_VOCAB_SIZE")
    enable_query_expansion: bool = Field(default=True, validation_alias="ENABLE_QUERY_EXPANSION")
    rag_authority_boost_official: float = Field(default=1.2, validation_alias="RAG_AUTHORITY_BOOST_OFFICIAL")
    rag_authority_boost_semi: float = Field(default=1.0, validation_alias="RAG_AUTHORITY_BOOST_SEMI")
    rag_authority_boost_third_party: float = Field(default=0.8, validation_alias="RAG_AUTHORITY_BOOST_THIRD_PARTY")
    rag_recency_penalty_max: float = Field(default=0.1, validation_alias="RAG_RECENCY_PENALTY_MAX")
    rag_recency_penalty_days: int = Field(default=365, validation_alias="RAG_RECENCY_PENALTY_DAYS")

    # ============================================
    # Reranker Configuration
    # ============================================
    reranker_api_type: Literal["mock", "cohere", "jina"] = Field(default="mock", validation_alias="RERANKER_API_TYPE")
    reranker_api_key: Optional[str] = Field(default=None, validation_alias="RERANKER_API_KEY")
    reranker_model_name: str = Field(default="rerank-english-v2.0", validation_alias="RERANKER_MODEL_NAME")

    # ============================================
    # MLflow Tracking
    # ============================================
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="german-visa-rag", validation_alias="MLFLOW_EXPERIMENT_NAME")
    enable_mlflow: bool = Field(default=True, validation_alias="ENABLE_MLFLOW")

    # ============================================
    # Observability
    # ============================================
    enable_structured_logging: bool = Field(default=True, validation_alias="ENABLE_STRUCTURED_LOGGING")
    enable_prometheus: bool = Field(default=False, validation_alias="ENABLE_PROMETHEUS")

    # ============================================
    # Paths
    # ============================================
    seed_urls_path: Path = Field(default=Path("src/ingestion/seed_urls.yml"), validation_alias="SEED_URLS_PATH")
    logs_dir: Path = Field(default=Path("./logs"), validation_alias="LOGS_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("sqlite_db_path", "logs_dir", mode="before")
    @classmethod
    def ensure_path_exists(cls, v):
        """Ensure directories exist."""
        p = Path(v)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("allowed_hosts", "allowed_origins", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v):
        """Parse comma-separated string into a list if necessary."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
