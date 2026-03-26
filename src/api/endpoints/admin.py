from fastapi import APIRouter, Depends
from src.logger import logger
from src.api.auth import auth
from src.ingestion.scheduler import get_scheduler
from src.storage.sqlite_state_store import get_state_store

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest/trigger")
async def trigger_ingestion(x_api_key: str = Depends(auth.verify_api_key)):
    """Manually trigger ingestion pipeline."""
    logger.info("Manual ingestion triggered by admin")
    scheduler = get_scheduler()
    await scheduler.trigger_manual_ingestion()
    return {"status": "ingestion_started"}


@router.get("/ingest/stats")
async def get_ingestion_stats(x_api_key: str = Depends(auth.verify_api_key)):
    """Get ingestion statistics."""
    state_store = get_state_store()
    return {"statistics": state_store.get_stats()}
