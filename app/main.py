"""
Main FastAPI application for Lacuna backend.
Handles CORS, request logging middleware, lifespan events, and router registration.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import close_db, init_db
from app.routers import brain, concepts, documents, health, rooms, search
from app.routers import pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup / shutdown helpers
# ---------------------------------------------------------------------------

async def _check_database() -> bool:
    """Initialise DB tables and verify the connection.  Returns True on success."""
    try:
        await init_db()
        logger.info("✓ Database connection OK")
        return True
    except Exception as exc:
        logger.error("✗ Database connection failed: %s", exc)
        raise


async def _check_ollama() -> dict:
    """
    Verify Ollama is reachable and check that the required models are available.
    Returns a dict with status info.  Never raises — warnings are logged instead.
    """
    result = {"reachable": False, "embed_model": False, "llm_model": False, "models": []}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
        if resp.status_code != 200:
            logger.warning("⚠ Ollama responded with status %d", resp.status_code)
            return result

        result["reachable"] = True
        available = [m["name"] for m in resp.json().get("models", [])]
        result["models"] = available
        logger.info("✓ Ollama reachable — available models: %s", available)

        # Check embed model (partial match so "nomic-embed-text:latest" still matches)
        embed_model = settings.OLLAMA_EMBED_MODEL
        result["embed_model"] = any(
            m == embed_model or m.startswith(embed_model.split(":")[0])
            for m in available
        )
        if result["embed_model"]:
            logger.info("  ✓ Embedding model '%s' is available", embed_model)
        else:
            logger.warning(
                "  ⚠ Embedding model '%s' not found — run: ollama pull %s",
                embed_model,
                embed_model,
            )

        llm_model = settings.OLLAMA_LLM_MODEL
        result["llm_model"] = any(
            m == llm_model or m.startswith(llm_model.split(":")[0])
            for m in available
        )
        if result["llm_model"]:
            logger.info("  ✓ LLM model '%s' is available", llm_model)
        else:
            logger.warning(
                "  ⚠ LLM model '%s' not found — run: ollama pull %s",
                llm_model,
                llm_model,
            )

    except Exception as exc:
        logger.error(
            "✗ Ollama unreachable (%s) — embedding and LLM features will fail", exc
        )
    return result


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown event handler."""
    logger.info("=" * 60)
    logger.info("  Starting Lacuna backend …")
    logger.info("=" * 60)

    # 1 — Database (required; raises on failure)
    await _check_database()

    # 2 — Ollama (optional; logs warnings but continues)
    ollama_status = await _check_ollama()
    if not ollama_status["reachable"]:
        logger.warning(
            "Ollama is not running.  Start it with: ollama serve\n"
            "  Embedding and LLM features will be unavailable until Ollama is up."
        )

    # 3 — Upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info("✓ Upload directory: %s", os.path.abspath(settings.UPLOAD_DIR))

    logger.info("=" * 60)
    logger.info("  Lacuna backend ready on http://%s:%d", settings.HOST, settings.PORT)
    logger.info("  Swagger UI : http://%s:%d/docs", settings.HOST, settings.PORT)
    logger.info("  Health     : http://%s:%d/api/health", settings.HOST, settings.PORT)
    logger.info("=" * 60)

    yield  # ← server is running

    logger.info("Shutting down Lacuna backend …")
    await close_db()
    logger.info("✓ Shutdown complete.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Lacuna API",
    description=(
        "**Lacuna** — AI-powered research discovery platform.\n\n"
        "Upload research papers (PDF/DOCX), extract concepts, detect "
        "relationships, identify knowledge gaps, and query your collection "
        "via RAG.\n\n"
        "Key endpoints:\n"
        "- `POST /api/documents/upload` — upload a document\n"
        "- `POST /api/documents/{id}/process` — embed + extract one document\n"
        "- `POST /api/concepts/build` — rebuild the full knowledge graph\n"
        "- `GET  /api/concepts/map` — React Flow concept map\n"
        "- `POST /api/brain/chat` — RAG chat\n"
        "- `POST /api/pipeline/process-all` — batch process + rebuild\n"
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every request with method, path, status code, and elapsed time.
    Attaches an ``X-Process-Time`` header (milliseconds) to every response.
    """
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

    # Skip noisy health-check polling from the frontend
    if request.url.path not in ("/api/health", "/"):
        logger.info(
            "%s %s → %d  (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

    response.headers["X-Process-Time"] = f"{elapsed_ms}ms"
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a structured JSON error for any unhandled exception."""
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router,     prefix="/api/health",    tags=["Health"])
app.include_router(rooms.router,     prefix="/api/rooms",     tags=["Rooms"])
app.include_router(documents.router,  prefix="/api/documents", tags=["Documents"])
app.include_router(concepts.router,   prefix="/api/concepts",  tags=["Concepts"])
app.include_router(brain.router,      prefix="/api/brain",     tags=["Brain"])
app.include_router(search.router,     prefix="/api/search",    tags=["Search"])
app.include_router(pipeline.router,   prefix="/api/pipeline",  tags=["Pipeline"])


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """API root — returns basic service info."""
    return {
        "name": "Lacuna API",
        "version": "0.2.0",
        "description": "Research Discovery Platform Backend",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "documents": "/api/documents",
            "concepts": "/api/concepts",
            "map": "/api/concepts/map",
            "brain": "/api/brain",
            "pipeline": "/api/pipeline",
            "search": "/api/search",
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
