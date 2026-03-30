import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from src.api.auth import auth
from src.ingestion.scheduler import get_scheduler
from src.logger import logger
from src.storage.sqlite_state_store import get_state_store

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest/trigger")
async def trigger_ingestion(
    request: Request,
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Manually trigger ingestion pipeline."""
    client_ip = request.client.host if request.client else "unknown"
    key_hint = hashlib.sha256(x_api_key.encode()).hexdigest()[:12]
    logger.info(
        "Admin action: trigger_ingestion | ip=%s key_hint=%s ts=%s",
        client_ip,
        key_hint,
        datetime.now(timezone.utc).isoformat(),
    )
    scheduler = get_scheduler()
    await scheduler.trigger_manual_ingestion()
    return {"status": "ingestion_started"}


@router.get("/ingest/stats")
async def get_ingestion_stats(
    request: Request,
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Get ingestion statistics."""
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        "Admin action: ingest_stats | ip=%s ts=%s",
        client_ip,
        datetime.now(timezone.utc).isoformat(),
    )
    state_store = get_state_store()
    return {"statistics": state_store.get_stats()}
