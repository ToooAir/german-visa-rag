"""
FastAPI application entry point.
Sets up middleware, routes, lifecycle handlers, and exception handlers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import time
import os

from src.config import settings
from src.logger import logger
from src.exceptions import RAGException
from src.storage.redis_cache import query_cache
from src.api.routes import router, query_router, admin_router
from src.ingestion.scheduler import get_scheduler
from src.vector_db.qdrant_client_wrapper import get_qdrant_client
import httpx
from src.ingestion.url_discoverer import URLDiscoverer
from src.rag.hybrid_retriever import HybridRetriever
from src.storage.sqlite_state_store import get_state_store
from src.api.rate_limiter import rate_limiter


# ============================================
# Lifecycle Handlers
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    
    # Startup
    logger.info(
        "Application starting",
        extra={
            "environment": settings.environment,
            "debug": settings.debug,
        }
    )
    
    # Initialize global HTTP client
    http_client = httpx.AsyncClient(
        timeout=settings.crawler_timeout_seconds,
        follow_redirects=True,
        limits=httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10,
        ),
    )
    app.state.http_client = http_client
    
    try:
        # Initialize dependencies
        state_store = get_state_store()
        app.state.url_discoverer = URLDiscoverer(client=http_client, state_store=state_store)
        qdrant = get_qdrant_client()
        app.state.hybrid_retriever = HybridRetriever(qdrant_client=qdrant)
        
        from src.rag.answer_generator import AnswerGenerator
        app.state.answer_generator = AnswerGenerator(retriever=app.state.hybrid_retriever)

        # Initialize Qdrant collection
        await qdrant.ensure_collection_exists()
        logger.info("Qdrant collection initialized")
        
        # Start scheduler (skip in cloud if using external cron)
        if settings.enable_internal_scheduler:
            scheduler = get_scheduler()
            scheduler.start()
            logger.info("Internal ingestion scheduler started")
        else:
            logger.info("Internal scheduler disabled. Awaiting external cron triggers.")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Application shutting down")
        
        try:
            await app.state.http_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")

        try:
            if settings.enable_internal_scheduler:
                scheduler = get_scheduler()
                await scheduler.shutdown()
        except Exception as e:
            logger.warning(f"Error during scheduler shutdown: {e}")
        
        try:
            qdrant = get_qdrant_client()
            await qdrant.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant: {e}")
            
        try:
            await query_cache.close()
        except Exception as e:
            logger.warning(f"Error closing Redis: {e}")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="German Visa & Chancenkarte RAG API",
    description="RAG-based Q&A system for German visa regulations",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle custom RAG domain exceptions."""
    logger.error(f"RAG Exception on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "rag_processing_error",
            "message": str(exc),
        },
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected global exceptions."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error", 
            "message": "An unexpected error occurred."
        },
    )


# ============================================
# Middleware
# ============================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZIP compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted hosts
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )


# ============================================
# Request/Response Logging
# ============================================

@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Enforce IP-based rate limiting on sensitive core endpoints."""
    # Only rate limit chat/query endpoints to avoid impacting health/admin/docs
    if request.url.path.startswith(("/query", "/ask")):
        try:
            await rate_limiter.check_rate_limit(request)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content=e.detail
            )
            
    return await call_next(request)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses with latency."""
    start_time = time.time()
    
    logger.debug(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "query": dict(request.query_params),
        }
    )
    
    try:
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.debug(
            f"{request.method} {request.url.path} - {response.status_code}",
            extra={
                "status_code": response.status_code,
                "latency_seconds": process_time,
            }
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        logger.error(f"Request processing failed: {e}", exc_info=True)
        raise


# ============================================
# Routes
# ============================================

app.include_router(router)
app.include_router(query_router)
app.include_router(admin_router)


# ============================================
# Frontend Static Files & SPA Routing
# ============================================

# Mount static files (JS, CSS, etc.)
# Note: We do this after routers to ensure API routes take precedence
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

if os.path.exists(static_dir):
    # Mount assets directory for JS/CSS
    assets_dir = os.path.join(static_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all route to serve the SPA index.html for all non-API paths."""
        # Check if the path is an API path, let it pass to routers or 404
        if full_path.startswith(("v1", "query", "ask", "docs", "openapi")):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "not_found", "message": f"Path /{full_path} not found"}
            )
        
        # 1. Check if it's a direct file in the static root (like vite.svg or favicon.ico)
        file_path = os.path.join(static_dir, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # 2. Otherwise, serve index.html (SPA routing)
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "frontend_not_built", "message": "Frontend assets not found"}
        )
else:
    logger.warning(f"Static directory not found at {static_dir}. Frontend will not be served.")


# ============================================
# Root & Docs Endpoints
# ============================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "German Visa & Chancenkarte RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }

@app.get("/docs/openai")
async def openai_docs():
    """OpenAI-compatible API documentation."""
    return {
        "description": "This API implements OpenAI-compatible endpoints",
        "endpoints": {
            "chat.completions": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "description": "OpenAI-compatible chat completions (supports streaming)",
                "auth": "X-API-Key header",
            }
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
