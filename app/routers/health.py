"""
Health check endpoint.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime
import logging

from app.database import get_db
from app.models.schemas import HealthCheckResponse
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthCheckResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint to verify system status.

    Returns:
        HealthCheckResponse with status of database and Bedrock
    """
    # Check database connection
    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "error"

    # Check AWS Bedrock connection
    bedrock_status = "ok"
    try:
        embedding_service = EmbeddingService()
        is_healthy = await embedding_service.check_bedrock_health()
        if not is_healthy:
            bedrock_status = "error"
    except Exception as e:
        logger.error(f"Bedrock health check failed: {e}")
        bedrock_status = "error"

    # Overall status
    overall_status = "healthy" if db_status == "ok" and bedrock_status == "ok" else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        database=db_status,
        ollama=bedrock_status,
        timestamp=datetime.utcnow()
    )
