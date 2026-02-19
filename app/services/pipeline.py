"""
Master pipeline orchestrator for the Lacuna knowledge graph.

Public API
----------
LacunaPipeline.process_document(document_id, db)
    → ProcessingResult
    Single-document pipeline: embed chunks → extract concepts/claims/relationships.

LacunaPipeline.rebuild_project_knowledge(project_id, db)
    → KnowledgeResult
    Full project rebuild: normalise → cluster → relationships → gaps → consensus.

LacunaPipeline.add_document_and_update(document_id, project_id, db)
    → FullResult
    Convenience wrapper: process_document + rebuild_project_knowledge.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models.database_models import Chunk, Document
from app.services.brain_service import BrainService
from app.services.clustering import ConceptClusterer
from app.services.embedding import OllamaEmbeddingService
from app.services.gap_detector import GapDetector
from app.services.llm_extractor import OllamaLLMService
from app.services.normalizer import ConceptNormalizer
from app.services.relationships import RelationshipDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ProcessingResult:
    """Result of processing a single document through the embed + extract pipeline."""

    document_id: int
    document_title: str
    embedded_chunks: int
    total_chunks: int
    chunks_processed: int
    chunks_skipped: int
    concepts_extracted: int
    concepts_saved: int
    claims_saved: int
    relationships_found: int
    errors: List[str]
    processing_time_seconds: float
    message: str


@dataclasses.dataclass
class KnowledgeResult:
    """Result of rebuilding the full project knowledge graph."""

    project_id: int
    # Normalisation
    concepts_before_normalize: int
    concepts_after_normalize: int
    merged_count: int
    # Clustering
    num_clusters: int
    algorithm_used: str
    clustered_concepts: int
    # Relationships
    relationships_found: int
    relationships_saved: int
    # Gaps
    expected_gaps: int
    bridging_gaps: int
    weak_coverage_gaps: int
    total_gaps: int
    # Brain / consensus
    concepts_scored: int
    strong_consensus_count: int
    contested_count: int
    contradiction_count: int
    brain_summary: str
    processing_time_seconds: float
    message: str


@dataclasses.dataclass
class FullResult:
    """Combined result of uploading + processing a document and rebuilding knowledge."""

    document_result: ProcessingResult
    knowledge_result: KnowledgeResult
    total_time_seconds: float
    message: str


# ---------------------------------------------------------------------------
# LacunaPipeline
# ---------------------------------------------------------------------------

class LacunaPipeline:
    """
    Master orchestrator that coordinates all Lacuna services.

    Each method is designed to be called from a FastAPI endpoint or background
    task.  Methods are stateless — they instantiate services internally so
    the class can be instantiated cheaply per request.
    """

    def __init__(self) -> None:
        self._embedder = OllamaEmbeddingService()
        self._llm = OllamaLLMService()
        self._normalizer = ConceptNormalizer()
        self._clusterer = ConceptClusterer()
        self._rel_detector = RelationshipDetector()
        self._gap_detector = GapDetector()
        self._brain_service = BrainService()

    # ------------------------------------------------------------------
    # Single-document pipeline
    # ------------------------------------------------------------------

    async def process_document(
        self, document_id: int, db: AsyncSession
    ) -> ProcessingResult:
        """
        Process a single document through the embed + extract pipeline.

        Steps
        -----
        1. Verify the document exists.
        2. Embed all un-embedded chunks (skips cache hits).
        3. Run LLM concept + claim + relationship extraction chunk-by-chunk.

        Normalisation and clustering happen at the project level via
        ``rebuild_project_knowledge`` and are intentionally separated so
        that the frontend can show per-document progress.
        """
        t0 = time.monotonic()

        doc_result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = doc_result.scalar_one_or_none()
        if document is None:
            raise ValueError(f"Document {document_id} not found.")

        logger.info(
            "Pipeline.process_document: id=%d (%s) — embedding phase",
            document_id,
            document.filename,
        )

        # Step 1 — embed
        try:
            embedded_chunks, total_chunks = await self._embedder.embed_document_chunks(
                document_id, db
            )
        except Exception as exc:
            logger.error(
                "Pipeline.process_document: embed failed for id=%d: %s",
                document_id,
                exc,
            )
            raise

        logger.info(
            "Pipeline.process_document: id=%d — extraction phase (%d/%d chunks embedded)",
            document_id,
            embedded_chunks,
            total_chunks,
        )

        # Step 2 — extract
        try:
            summary = await self._llm.process_document(document_id, db)
        except Exception as exc:
            logger.error(
                "Pipeline.process_document: extraction failed for id=%d: %s",
                document_id,
                exc,
            )
            raise

        elapsed = round(time.monotonic() - t0, 2)
        message = (
            f"Document '{document.filename}' processed in {elapsed}s: "
            f"embedded {embedded_chunks}/{total_chunks} chunks, "
            f"saved {summary.concepts_saved} concepts and {summary.claims_saved} claims."
        )
        if summary.errors:
            message += f" {len(summary.errors)} chunk(s) had errors."

        logger.info("Pipeline.process_document: %s", message)

        return ProcessingResult(
            document_id=document_id,
            document_title=document.filename,
            embedded_chunks=embedded_chunks,
            total_chunks=total_chunks,
            chunks_processed=summary.chunks_processed,
            chunks_skipped=summary.chunks_skipped,
            concepts_extracted=summary.concepts_extracted,
            concepts_saved=summary.concepts_saved,
            claims_saved=summary.claims_saved,
            relationships_found=summary.relationships_found,
            errors=summary.errors,
            processing_time_seconds=elapsed,
            message=message,
        )

    # ------------------------------------------------------------------
    # Project-level knowledge rebuild
    # ------------------------------------------------------------------

    async def rebuild_project_knowledge(
        self, project_id: int, db: AsyncSession
    ) -> KnowledgeResult:
        """
        Rebuild the entire knowledge graph for a project.

        Steps (all synchronous, committed incrementally)
        -------------------------------------------------
        1. Normalise concepts — exact-name dedup + AgglomerativeClustering
           merge; remaps claims/relationships to canonical IDs.
        2. Cluster — HDBSCAN (KMeans fallback); builds 2-level hierarchy and
           detects cross-cluster bridge pairs.
        3. Detect relationships — embedding similarity + LLM classification
           + multi-signal strength scoring.
        4. Detect gaps — expected topics (LLM), bridging gaps (graph),
           weak-coverage concepts (statistical).
        5. Build consensus scores — per-concept support/contradict weighting;
           generates LLM synthesis summary; persists BrainState.

        Each step is wrapped in a try/except so a partial failure does not
        abort the whole pipeline — the failing step is logged and zeros are
        reported for that stage.
        """
        t0 = time.monotonic()
        logger.info(
            "Pipeline.rebuild_project_knowledge: starting for project id=%d", project_id
        )

        # ---- Step 1: Normalise ----
        logger.info("Pipeline: [1/5] normalising concepts …")
        try:
            norm = await self._normalizer.normalize_project(project_id, db)
            concepts_before = norm.total_concepts_before
            concepts_after = norm.canonical_concepts_after
            merged_count = norm.merged_count
            logger.info(
                "Pipeline: [1/5] normalised %d → %d concepts (%d merged)",
                concepts_before,
                concepts_after,
                merged_count,
            )
        except Exception as exc:
            logger.error("Pipeline: [1/5] normalise failed: %s", exc, exc_info=True)
            concepts_before = concepts_after = merged_count = 0

        # ---- Step 2: Cluster ----
        logger.info("Pipeline: [2/5] clustering concepts …")
        num_clusters = 0
        algorithm_used = "none"
        clustered_concepts = 0
        try:
            cluster = await self._clusterer.cluster_project(project_id, db)
            num_clusters = cluster.num_clusters
            algorithm_used = cluster.algorithm_used
            clustered_concepts = cluster.clustered_concepts
            logger.info(
                "Pipeline: [2/5] %d clusters (%s), %d concepts clustered",
                num_clusters,
                algorithm_used,
                clustered_concepts,
            )
        except Exception as exc:
            logger.warning("Pipeline: [2/5] cluster failed (continuing): %s", exc)

        # ---- Step 3: Relationships ----
        logger.info("Pipeline: [3/5] detecting relationships …")
        relationships_found = relationships_saved = 0
        try:
            rel = await self._rel_detector.detect_relationships(project_id, db)
            relationships_found = rel.relationships_found
            relationships_saved = rel.relationships_saved
            logger.info(
                "Pipeline: [3/5] %d relationships found, %d saved",
                relationships_found,
                relationships_saved,
            )
        except Exception as exc:
            logger.warning("Pipeline: [3/5] relationship detection failed (continuing): %s", exc)

        # ---- Step 4: Gaps ----
        logger.info("Pipeline: [4/5] detecting knowledge gaps …")
        expected_gaps = bridging_gaps = weak_coverage_gaps = total_gaps = 0
        try:
            gap = await self._gap_detector.detect_gaps(project_id, db)
            expected_gaps = gap.expected_gaps_count
            bridging_gaps = gap.bridging_gaps_count
            weak_coverage_gaps = gap.weak_coverage_count
            total_gaps = gap.total_gaps
            logger.info(
                "Pipeline: [4/5] %d gaps total (%d expected, %d bridging, %d weak)",
                total_gaps,
                expected_gaps,
                bridging_gaps,
                weak_coverage_gaps,
            )
        except Exception as exc:
            logger.warning("Pipeline: [4/5] gap detection failed (continuing): %s", exc)

        # ---- Step 5: Brain / consensus ----
        logger.info("Pipeline: [5/5] building brain state …")
        concepts_scored = strong_count = contested_count = contradiction_count = 0
        brain_summary = ""
        try:
            brain = await self._brain_service.build_brain(
                project_id, db, clear_existing=False
            )
            concepts_scored = brain.concepts_scored
            strong_count = brain.strong_consensus_count
            contested_count = brain.contested_count
            contradiction_count = brain.contradiction_count
            brain_summary = brain.summary_text
            logger.info(
                "Pipeline: [5/5] brain built — %d concepts scored, "
                "%d strong / %d contested / %d contradicted",
                concepts_scored,
                strong_count,
                contested_count,
                contradiction_count,
            )
        except Exception as exc:
            logger.warning("Pipeline: [5/5] brain build failed (continuing): %s", exc)

        elapsed = round(time.monotonic() - t0, 2)
        message = (
            f"Knowledge rebuilt for project {project_id} in {elapsed}s: "
            f"{concepts_after} concepts ({merged_count} merged), "
            f"{num_clusters} clusters ({algorithm_used}), "
            f"{relationships_saved} relationships, "
            f"{total_gaps} gaps."
        )
        logger.info("Pipeline.rebuild_project_knowledge: %s", message)

        return KnowledgeResult(
            project_id=project_id,
            concepts_before_normalize=concepts_before,
            concepts_after_normalize=concepts_after,
            merged_count=merged_count,
            num_clusters=num_clusters,
            algorithm_used=algorithm_used,
            clustered_concepts=clustered_concepts,
            relationships_found=relationships_found,
            relationships_saved=relationships_saved,
            expected_gaps=expected_gaps,
            bridging_gaps=bridging_gaps,
            weak_coverage_gaps=weak_coverage_gaps,
            total_gaps=total_gaps,
            concepts_scored=concepts_scored,
            strong_consensus_count=strong_count,
            contested_count=contested_count,
            contradiction_count=contradiction_count,
            brain_summary=brain_summary,
            processing_time_seconds=elapsed,
            message=message,
        )

    # ------------------------------------------------------------------
    # Combined: process document + rebuild knowledge
    # ------------------------------------------------------------------

    async def add_document_and_update(
        self,
        document_id: int,
        project_id: int,
        db: AsyncSession,
    ) -> FullResult:
        """
        Full end-to-end pipeline for a freshly uploaded document.

        1. Embed + extract the document (``process_document``).
        2. Rebuild the project knowledge graph (``rebuild_project_knowledge``).

        The document must already exist in the database (uploaded via
        ``POST /api/documents/upload``) before calling this method.
        """
        t0 = time.monotonic()
        logger.info(
            "Pipeline.add_document_and_update: doc id=%d, project id=%d",
            document_id,
            project_id,
        )

        doc_result = await self.process_document(document_id, db)
        knowledge_result = await self.rebuild_project_knowledge(project_id, db)

        total_time = round(time.monotonic() - t0, 2)
        message = (
            f"Full pipeline complete in {total_time}s: "
            f"processed '{doc_result.document_title}', "
            f"knowledge graph rebuilt with {knowledge_result.num_clusters} cluster(s) "
            f"and {knowledge_result.relationships_saved} relationship(s)."
        )
        logger.info("Pipeline.add_document_and_update: %s", message)

        return FullResult(
            document_result=doc_result,
            knowledge_result=knowledge_result,
            total_time_seconds=total_time,
            message=message,
        )

    # ------------------------------------------------------------------
    # Preflight check
    # ------------------------------------------------------------------

    async def _preflight_check(self) -> None:
        """
        Verify Ollama is reachable before starting the pipeline.

        Raises ``RuntimeError`` if Ollama is down — callers should catch
        this and fail fast with a clear error message.
        """
        healthy = await self._embedder.check_ollama_health()
        if not healthy:
            raise RuntimeError(
                "Ollama is not reachable at "
                f"{self._embedder.base_url}. "
                "Start Ollama before running the pipeline."
            )
        logger.info("Preflight check passed: Ollama is reachable")

    # ------------------------------------------------------------------
    # Phased batch pipeline (background task)
    # ------------------------------------------------------------------

    async def process_all_phased(
        self,
        project_id: int,
        status: "PipelineStatus",
        document_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Three-phase pipeline: embed → extract → rebuild knowledge.

        If *document_ids* is provided, only those documents are embedded and
        extracted (incremental mode — used when new papers are added to a room
        that already has processed documents).  If ``None``, all documents in
        the project are processed.

        The knowledge rebuild (normalize → cluster → relationships → gaps →
        brain) **always** runs across the entire project.

        Designed to be run as an ``asyncio.Task`` via ``PipelineManager``.
        The *status* object is mutated in-place so callers can poll progress.
        """
        from app.services.pipeline_manager import PipelinePhase

        # Preflight: verify Ollama is reachable before doing any work
        try:
            await self._preflight_check()
        except RuntimeError as exc:
            logger.error("process_all_phased: preflight failed — %s", exc)
            status.phase = PipelinePhase.FAILED
            status.errors.append(str(exc))
            return

        # Load documents to process
        async with AsyncSessionLocal() as db:
            stmt = select(Document).where(Document.project_id == project_id)
            if document_ids is not None:
                stmt = stmt.where(Document.id.in_(document_ids))
            docs_result = await db.execute(stmt)
            documents = list(docs_result.scalars().all())

        status.total_documents = len(documents)

        if not documents:
            logger.info("process_all_phased: no documents to process for project %d", project_id)
            # Still run knowledge rebuild if there are other docs in the project
            if document_ids is not None:
                # Skip straight to knowledge rebuild
                pass
            else:
                status.phase = PipelinePhase.COMPLETED
                return

        # ── Phase 1: Embed documents ──────────────────────────────────
        # embed_document_chunks already skips chunks with existing embeddings,
        # but we skip the whole doc if all chunks are already embedded.
        if documents:
            status.phase = PipelinePhase.EMBEDDING
            for doc in documents:
                async with AsyncSessionLocal() as session:
                    try:
                        # Resume: check if doc already fully embedded
                        unembedded = await session.execute(
                            select(func.count(Chunk.id)).where(
                                Chunk.document_id == doc.id,
                                Chunk.embedding.is_(None),
                            )
                        )
                        if unembedded.scalar() == 0:
                            logger.info(
                                "process_all_phased: doc %d already fully embedded — skipping",
                                doc.id,
                            )
                            status.documents_embedded += 1
                            continue

                        status.current_document = doc.filename
                        await self._embedder.embed_document_chunks(doc.id, session)
                        status.documents_embedded += 1
                    except Exception as exc:
                        status.documents_failed += 1
                        status.errors.append(f"embed {doc.filename}: {str(exc)[:120]}")
                        logger.error("process_all_phased: embed failed doc %d: %s", doc.id, exc)

        # ── Phase 2: Extract documents (LLM) ─────────────────────────
        # process_document() already skips extracted/skipped chunks internally,
        # but we skip the whole doc if all chunks are already done.
        if documents:
            status.phase = PipelinePhase.EXTRACTING
            for doc in documents:
                async with AsyncSessionLocal() as session:
                    try:
                        # Resume: check if doc has any unprocessed chunks
                        unextracted = await session.execute(
                            select(func.count(Chunk.id)).where(
                                Chunk.document_id == doc.id,
                                Chunk.extraction_status.is_(None),
                            )
                        )
                        if unextracted.scalar() == 0:
                            logger.info(
                                "process_all_phased: doc %d already fully extracted — skipping",
                                doc.id,
                            )
                            status.documents_extracted += 1
                            continue

                        status.current_document = doc.filename
                        await self._llm.process_document(doc.id, session)
                        status.documents_extracted += 1
                    except Exception as exc:
                        status.documents_failed += 1
                        status.errors.append(f"extract {doc.filename}: {str(exc)[:120]}")
                        logger.error("process_all_phased: extract failed doc %d: %s", doc.id, exc)

        # ── Phase 3: Rebuild knowledge (5 steps) ─────────────────────
        status.current_document = None
        knowledge_steps = [
            (PipelinePhase.NORMALIZING, self._normalizer.normalize_project),
            (PipelinePhase.CLUSTERING, self._clusterer.cluster_project),
            (PipelinePhase.RELATIONSHIPS, self._rel_detector.detect_relationships),
            (PipelinePhase.GAPS, self._gap_detector.detect_gaps),
            (PipelinePhase.BRAIN, lambda pid, db: self._brain_service.build_brain(pid, db, clear_existing=False)),
        ]

        for phase_enum, step_fn in knowledge_steps:
            status.phase = phase_enum
            async with AsyncSessionLocal() as session:
                try:
                    await step_fn(project_id, session)
                except Exception as exc:
                    status.errors.append(f"{phase_enum.value}: {str(exc)[:120]}")
                    logger.error(
                        "process_all_phased: %s failed for project %d: %s",
                        phase_enum.value, project_id, exc,
                    )

        status.phase = PipelinePhase.COMPLETED
        logger.info(
            "process_all_phased: completed for project %d — "
            "%d embedded, %d extracted, %d failed in %.1fs",
            project_id,
            status.documents_embedded,
            status.documents_extracted,
            status.documents_failed,
            status.elapsed_seconds,
        )
