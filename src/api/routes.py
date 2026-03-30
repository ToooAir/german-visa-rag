"""
Main router aggregator that includes sub-routers from endpoints.
"""

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.endpoints.admin import router as admin_router_module
from src.api.endpoints.chat import router as chat_router
from src.api.endpoints.health import router as health_router
from src.api.endpoints.rag import router as rag_router
from src.logger import logger

# Assembler Router to match original structure
router = APIRouter()
router.include_router(chat_router)
router.include_router(health_router)

query_router = rag_router
admin_router = admin_router_module


# Exception Handlers
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning("HTTP error: %s - %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status": exc.status_code},
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
