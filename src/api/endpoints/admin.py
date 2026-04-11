import hashlib
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from src.api.auth import auth
from src.ingestion.scheduler import get_scheduler
from src.logger import logger
from src.storage.sqlite_state_store import get_state_store

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest/trigger")
async def trigger_ingestion(
    request: Request,
    force: bool = Query(
        False, description="Re-process all documents even if content hasn't changed (equivalent to CLI --force)"
    ),
    force_discover: bool = Query(
        False,
        description="Force fresh URL discovery, bypassing the visited-URL cache (equivalent to CLI --force-discover)",
    ),
    auto_discover: Optional[bool] = Query(
        None, description="Override CRAWLER_DISCOVERY_ENABLED for this run only. If omitted, uses the env setting"
    ),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Manually trigger the full ingestion pipeline.

    - **force**: Re-process all documents even if content hasn't changed (CLI `--force`).
    - **force_discover**: Bypass visited-URL cache and re-crawl all discovery paths (CLI `--force-discover`).
    - **auto_discover**: Override discovery mode for this run only (`true` = discovery, `false` = seed URLs only).
      Omit to use the server's `CRAWLER_DISCOVERY_ENABLED` setting.
    """
    client_ip = request.client.host if request.client else "unknown"
    key_hint = hashlib.sha256(x_api_key.encode()).hexdigest()[:12]
    logger.info(
        "Admin action: trigger_ingestion | ip=%s key_hint=%s force=%s force_discover=%s auto_discover=%s ts=%s",
        client_ip,
        key_hint,
        force,
        force_discover,
        auto_discover,
        datetime.now(timezone.utc).isoformat(),
    )
    scheduler = get_scheduler()
    result = await scheduler.trigger_manual_ingestion(
        force=force,
        force_discover=force_discover,
        auto_discover=auto_discover,
    )
    return {
        "status": "ingestion_started",
        "force": force,
        "force_discover": force_discover,
        "mode": result["mode"],
    }


@router.post("/ingest/single")
async def ingest_single_url(
    request: Request,
    url: str = Query(..., description="URL to ingest (equivalent to CLI --source)"),
    force: bool = Query(False, description="Re-process even if this URL was already ingested"),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Ingest a single URL immediately.

    Equivalent to: `python -m src.ingestion.cli ingest --source <url> [--force]`
    """
    client_ip = request.client.host if request.client else "unknown"
    key_hint = hashlib.sha256(x_api_key.encode()).hexdigest()[:12]
    logger.info(
        "Admin action: ingest_single | ip=%s key_hint=%s url=%s force=%s ts=%s",
        client_ip,
        key_hint,
        url,
        force,
        datetime.now(timezone.utc).isoformat(),
    )
    scheduler = get_scheduler()
    result = await scheduler.ingest_single_url(url=url, force=force)
    return {"status": "ingestion_completed", "url": url, "force": force, "result": result}


@router.post("/discover")
async def discover_urls(
    request: Request,
    domain: Optional[str] = Query(
        None,
        description="Limit discovery to a specific domain (e.g. make-it-in-germany.com). Omit to discover all configured domains",
    ),
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Dry-run URL discovery — returns URLs that would be crawled without ingesting anything.

    Equivalent to: `python -m src.ingestion.cli discover [--domain <domain>]`
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        "Admin action: discover_urls | ip=%s domain=%s ts=%s",
        client_ip,
        domain or "all",
        datetime.now(timezone.utc).isoformat(),
    )
    scheduler = get_scheduler()
    results = await scheduler.discover_urls(domain=domain)
    total = sum(r["total"] for r in results)
    return {"status": "discovery_completed", "total_urls": total, "domains": results}


@router.get("/ingest/stats")
async def get_ingestion_stats(
    request: Request,
    x_api_key: str = Depends(auth.verify_api_key),
):
    """Get ingestion statistics from the SQLite state store."""
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        "Admin action: ingest_stats | ip=%s ts=%s",
        client_ip,
        datetime.now(timezone.utc).isoformat(),
    )
    state_store = get_state_store()
    return {"statistics": state_store.get_stats()}
