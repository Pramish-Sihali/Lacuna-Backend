"""
Main FastAPI application for Lacuna backend.
Handles CORS, lifespan events, and router registration.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import httpx
from datetime import datetime

from app.config import settings
from app.database import init_db, close_db
from app.routers import health, documents, concepts, brain, search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    Tests connections to Ollama and PostgreSQL on startup.
    """
    logger.info("Starting Lacuna backend...")

    # Test database connection
    try:
        await init_db()
        logger.info("✓ Database connection successful")
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        raise

    # Test Ollama connection
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                logger.info("✓ Ollama connection successful")
                models = response.json()
                logger.info(f"Available Ollama models: {[m['name'] for m in models.get('models', [])]}")
            else:
                logger.warning(f"⚠ Ollama responded with status {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Ollama connection failed: {e}")
        logger.warning("Some features may not work without Ollama running")

    # Create upload directory if it doesn't exist
    import os
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"✓ Upload directory ready: {settings.UPLOAD_DIR}")

    logger.info("✓ Lacuna backend started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Lacuna backend...")
    await close_db()
    logger.info("✓ Lacuna backend shut down")


# Create FastAPI app
app = FastAPI(
    title="Lacuna API",
    description="Backend API for Lacuna - Research Discovery Platform",
    version="0.1.0",
    lifespan=lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions globally."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# Include routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(concepts.router, prefix="/api/concepts", tags=["Concepts"])
app.include_router(brain.router, prefix="/api/brain", tags=["Brain"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Lacuna API",
        "version": "0.1.0",
        "description": "Research Discovery Platform Backend",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
