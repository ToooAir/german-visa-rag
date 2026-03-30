from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.auth import auth
from src.logger import logger
from src.storage.redis_cache import query_cache
from src.storage.sqlite_state_store import get_state_store
from src.vector_db.qdrant_client_wrapper import get_qdrant_client

router = APIRouter(prefix="/v1", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    dependencies: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        qdrant = get_qdrant_client()
        qdrant_ok = await qdrant.health_check()

        state_store = get_state_store()
        db_ok = state_store.db_path.exists()

        # Ping Redis if enabled; treat missing client as degraded (not failed)
        redis_ok = True
        if query_cache.enabled and query_cache.redis:
            try:
                await query_cache.redis.ping()
            except Exception:
                redis_ok = False

        dependencies = {
            "qdrant": "✓ OK" if qdrant_ok else "✗ FAILED",
            "sqlite": "✓ OK" if db_ok else "✗ FAILED",
            "redis": "✓ OK" if redis_ok else "✗ FAILED",
        }

        all_ok = all("✓" in v for v in dependencies.values())

        return HealthResponse(
            status="healthy" if all_ok else "degraded",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="0.1.0",
            dependencies=dependencies,
        )

    except Exception as e:
        logger.error("Health check failed: %s", e)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="0.1.0",
            dependencies={"error": str(e)},
        )


@router.get("/stats")
async def get_stats(x_api_key: str = Depends(auth.verify_api_key)):
    """Get ingestion and query statistics."""
    state_store = get_state_store()
    stats = state_store.get_stats()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "statistics": stats,
    }
