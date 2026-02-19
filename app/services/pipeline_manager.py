"""
In-memory singleton that tracks background pipeline tasks per project.

Usage
-----
    from app.services.pipeline_manager import pipeline_manager, PipelineStatus

    status = pipeline_manager.start(project_id, some_coroutine(project_id, status))
    # ... later ...
    current = pipeline_manager.get_status(project_id)
"""
from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import time
from typing import Any, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline phase enum
# ---------------------------------------------------------------------------

class PipelinePhase(str, enum.Enum):
    QUEUED = "queued"
    EMBEDDING = "embedding"
    EXTRACTING = "extracting"
    NORMALIZING = "normalizing"
    CLUSTERING = "clustering"
    RELATIONSHIPS = "relationships"
    GAPS = "gaps"
    BRAIN = "brain"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Pipeline status (mutable dataclass shared between task and poller)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PipelineStatus:
    project_id: int
    phase: PipelinePhase = PipelinePhase.QUEUED
    total_documents: int = 0
    documents_embedded: int = 0
    documents_extracted: int = 0
    documents_failed: int = 0
    current_document: Optional[str] = None
    errors: List[str] = dataclasses.field(default_factory=list)
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    # Optional: restrict processing to specific document IDs (incremental mode)
    document_ids: Optional[List[int]] = None

    @property
    def elapsed_seconds(self) -> float:
        end = self.completed_at if self.completed_at else time.monotonic()
        return round(end - self.started_at, 2)


# ---------------------------------------------------------------------------
# Pipeline manager (class-level state — acts as a singleton)
# ---------------------------------------------------------------------------

class PipelineManager:
    """Manages background pipeline asyncio.Tasks per project."""

    _tasks: Dict[int, asyncio.Task] = {}
    _status: Dict[int, PipelineStatus] = {}

    @classmethod
    def is_running(cls, project_id: int) -> bool:
        task = cls._tasks.get(project_id)
        return task is not None and not task.done()

    @classmethod
    def get_status(cls, project_id: int) -> Optional[PipelineStatus]:
        return cls._status.get(project_id)

    @classmethod
    def start(
        cls,
        project_id: int,
        coro: Coroutine[Any, Any, Any],
        status: Optional[PipelineStatus] = None,
    ) -> PipelineStatus:
        """
        Launch a background pipeline task for *project_id*.

        If *status* is provided (pre-created by the caller so it could be
        passed into the coroutine before this method is called), it is
        registered as-is.  Otherwise a fresh PipelineStatus is created.

        Returns the PipelineStatus object (shared with the running task so
        fields update in real time).
        """
        if cls.is_running(project_id):
            raise RuntimeError(f"Pipeline already running for project {project_id}")

        if status is None:
            status = PipelineStatus(project_id=project_id)
        cls._status[project_id] = status

        async def _wrapper() -> None:
            try:
                await coro
            except Exception as exc:
                logger.error(
                    "Pipeline task failed for project %d: %s", project_id, exc, exc_info=True
                )
                status.phase = PipelinePhase.FAILED
                status.errors.append(f"pipeline crash: {str(exc)[:200]}")
            finally:
                status.completed_at = time.monotonic()
                if status.phase not in (PipelinePhase.COMPLETED, PipelinePhase.FAILED):
                    status.phase = PipelinePhase.FAILED

        task = asyncio.create_task(_wrapper())
        cls._tasks[project_id] = task

        # Cleanup reference when done
        task.add_done_callback(lambda _t: cls._cleanup(project_id))

        logger.info("Pipeline task started for project %d", project_id)
        return status

    @classmethod
    def _cleanup(cls, project_id: int) -> None:
        """Remove the task reference (status is kept for polling)."""
        cls._tasks.pop(project_id, None)


# Module-level singleton instance
pipeline_manager = PipelineManager
