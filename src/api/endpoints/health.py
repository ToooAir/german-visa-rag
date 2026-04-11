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
    """Public health check response — no dependency details."""

    status: str
    timestamp: str
    version: str


class HealthDetailedResponse(BaseModel):
    """Detailed health check response — requires authentication."""

    status: str
    timestamp: str
    version: str
    dependencies: dict[str, str]


async def _check_dependencies() -> tuple[dict[str, str], bool]:
    """Run all dependency checks and return (results, all_ok)."""
    qdrant = get_qdrant_client()
    qdrant_ok = await qdrant.health_check()

    state_store = get_state_store()
    db_ok = state_store.db_path.exists()

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
    return dependencies, all_ok


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Public liveness probe — returns status only, no dependency details.

    Safe for use by uptime monitors and load balancers without authentication.
    For full dependency status, use GET /v1/health/detailed (requires X-API-Key).
    """
    try:
        _, all_ok = await _check_dependencies()
        return HealthResponse(
            status="healthy" if all_ok else "degraded",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="0.1.0",
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="0.1.0",
        )


@router.get("/health/detailed", response_model=HealthDetailedResponse)
async def health_check_detailed(x_api_key: str = Depends(auth.verify_api_key)):
    """Detailed health check — includes per-dependency status. Requires X-API-Key."""
    try:
        dependencies, all_ok = await _check_dependencies()
        return HealthDetailedResponse(
            status="healthy" if all_ok else "degraded",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="0.1.0",
            dependencies=dependencies,
        )
    except Exception as e:
        logger.error("Detailed health check failed: %s", e)
        return HealthDetailedResponse(
            status="unhealthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="0.1.0",
            dependencies={"error": str(e)},
        )
