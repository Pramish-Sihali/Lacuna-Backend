"""
Room management endpoints.

A "room" in the frontend maps to a "project" in the backend.
These endpoints provide CRUD for user-scoped rooms and room-scoped
versions of document/concept/brain operations.

Route summary
-------------
POST   /api/rooms                              — create room
GET    /api/rooms                              — list user's rooms
GET    /api/rooms/{room_id}                    — room detail
DELETE /api/rooms/{room_id}                    — delete room (cascades)

POST   /api/rooms/{room_id}/documents/upload   — upload document to room
GET    /api/rooms/{room_id}/documents/         — list room documents
DELETE /api/rooms/{room_id}/documents/{doc_id} — delete document
POST   /api/rooms/{room_id}/documents/{doc_id}/process — process document
POST   /api/rooms/{room_id}/documents/embed-all        — embed all docs

GET    /api/rooms/{room_id}/concepts/map       — concept map for room
POST   /api/rooms/{room_id}/concepts/build     — rebuild knowledge graph

POST   /api/rooms/{room_id}/brain/chat         — RAG chat scoped to room
GET    /api/rooms/{room_id}/brain/status       — brain status for room

POST   /api/rooms/{room_id}/pipeline/process-all — batch process + rebuild
"""
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal, get_db
from app.dependencies.auth import (
    get_authorized_project,
    get_current_user_id,
    get_or_create_user,
)
from app.models.database_models import (
    BrainState,
    Chunk,
    Claim,
    Concept,
    Document,
    Project,
    Relationship,
    User,
)
from app.models.schemas import (
    BrainChatRequest,
    BrainChatResponse,
    BrainStatusResponse,
    DocumentResponse,
    DocumentUploadResponse,
    EmbedAllResponse,
    GapItem,
    GapTypeSchema,
    KnowledgeBuildResponse,
    PipelineProcessAllResponse,
    PipelineStartRequest,
    PipelineStartResponse,
    PipelineStatusResponse,
    ProcessDocumentResponse,
    ReactFlowConceptMapResponse,
    ReactFlowEdge,
    ReactFlowEdgeData,
    ReactFlowMapMetadata,
    ReactFlowNode,
    ReactFlowNodeData,
    ReactFlowPosition,
    RoomCreateRequest,
    RoomResponse,
)
from app.services.brain_service import BrainService
from app.services.chunking import ChunkingService
from app.services.document_parser import DocumentParser
from app.services.embedding import OllamaEmbeddingService
from app.services.llm_extractor import OllamaLLMService
from app.services.pipeline import LacunaPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as exc:
        logger.warning("Could not remove file %r: %s", path, exc)


# ═══════════════════════════════════════════════════════════════════════════════
# ROOM CRUD
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("", response_model=RoomResponse, status_code=status.HTTP_201_CREATED)
async def create_room(
    body: RoomCreateRequest,
    user: User = Depends(get_or_create_user),
    db: AsyncSession = Depends(get_db),
) -> RoomResponse:
    """Create a new room for the authenticated user."""
    project = Project(
        name=body.name,
        description=body.description,
        user_id=user.id,
        color_index=body.color_index,
    )
    db.add(project)
    await db.flush()

    logger.info("Created room id=%d name=%r for user=%s", project.id, project.name, user.id)

    return RoomResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        color_index=project.color_index,
        paper_count=0,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.get("", response_model=List[RoomResponse])
async def list_rooms(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> List[RoomResponse]:
    """List all rooms belonging to the authenticated user, newest first."""
    result = await db.execute(
        select(Project)
        .where(Project.user_id == user_id)
        .order_by(Project.updated_at.desc())
    )
    projects = result.scalars().all()

    # Batch-fetch document counts
    project_ids = [p.id for p in projects]
    doc_counts: Dict[int, int] = {}
    if project_ids:
        dc_result = await db.execute(
            select(Document.project_id, func.count(Document.id).label("cnt"))
            .where(Document.project_id.in_(project_ids))
            .group_by(Document.project_id)
        )
        doc_counts = {row.project_id: row.cnt for row in dc_result}

    return [
        RoomResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            color_index=p.color_index or 0,
            paper_count=doc_counts.get(p.id, 0),
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in projects
    ]


@router.get("/{room_id}", response_model=RoomResponse)
async def get_room(
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> RoomResponse:
    """Get room details."""
    dc_result = await db.execute(
        select(func.count(Document.id))
        .where(Document.project_id == project.id)
    )
    paper_count = dc_result.scalar() or 0

    return RoomResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        color_index=project.color_index or 0,
        paper_count=paper_count,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.delete(
    "/{room_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_room(
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a room and all its data (documents, concepts, etc.)."""
    # Delete uploaded files from disk
    doc_result = await db.execute(
        select(Document.file_path).where(Document.project_id == project.id)
    )
    for (file_path,) in doc_result.all():
        _safe_remove(file_path)

    await db.delete(project)
    await db.flush()
    logger.info("Deleted room id=%d name=%r", project.id, project.name)


# ═══════════════════════════════════════════════════════════════════════════════
# ROOM-SCOPED DOCUMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/{room_id}/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def room_upload_document(
    project: Project = Depends(get_authorized_project),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """Upload a document to a specific room."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload must include a filename.")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Accepted: {', '.join(settings.SUPPORTED_FILE_TYPES)}",
        )

    stored_name = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, stored_name)
    file_size = 0

    try:
        async with aiofiles.open(file_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > settings.MAX_FILE_SIZE:
                    await out.close()
                    _safe_remove(file_path)
                    raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE // (1024*1024)} MB limit.")
                await out.write(chunk)

        parser = DocumentParser()
        try:
            parsed_doc = await parser.parse_document(file_path, file_ext)
        except RuntimeError as exc:
            _safe_remove(file_path)
            raise HTTPException(status_code=422, detail=str(exc))

        if not parsed_doc.full_text.strip():
            _safe_remove(file_path)
            raise HTTPException(status_code=422, detail="Document contains no extractable text.")

        document = Document(
            project_id=project.id,
            filename=file.filename,
            file_path=file_path,
            file_type=file_ext.lstrip("."),
            content_text=parsed_doc.full_text,
            metadata_json=parsed_doc.metadata,
        )
        db.add(document)
        await db.flush()

        chunker = ChunkingService()
        chunks = await chunker.chunk_document(parsed_doc, metadata={"filename": file.filename})
        for chunk_data in chunks:
            db.add(Chunk(
                document_id=document.id,
                content=chunk_data["content"],
                chunk_index=chunk_data["chunk_index"],
                embedding=None,
                metadata_json=chunk_data.get("metadata"),
            ))

        await db.flush()
        logger.info("Room %d: uploaded %r as doc id=%d (%d chunks)", project.id, file.filename, document.id, len(chunks))

        return DocumentUploadResponse(
            id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            status="processed",
            message=f"Document uploaded and parsed. {len(chunks)} chunks created.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Room upload error for %r", file.filename)
        _safe_remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {exc}")


@router.get("/{room_id}/documents/", response_model=List[DocumentResponse])
async def room_list_documents(
    project: Project = Depends(get_authorized_project),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> List[DocumentResponse]:
    """List all documents in a room."""
    docs_result = await db.execute(
        select(Document)
        .where(Document.project_id == project.id)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = docs_result.scalars().all()

    doc_ids = [d.id for d in documents]
    counts: Dict[int, int] = {}
    if doc_ids:
        counts_result = await db.execute(
            select(Chunk.document_id, func.count(Chunk.id).label("cnt"))
            .where(Chunk.document_id.in_(doc_ids))
            .group_by(Chunk.document_id)
        )
        counts = {row.document_id: row.cnt for row in counts_result}

    return [
        DocumentResponse(
            id=doc.id,
            project_id=doc.project_id,
            filename=doc.filename,
            file_type=doc.file_type,
            content_text=doc.content_text,
            metadata_json=doc.metadata_json,
            created_at=doc.created_at,
            chunk_count=counts.get(doc.id, 0),
        )
        for doc in documents
    ]


@router.delete(
    "/{room_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def room_delete_document(
    document_id: int,
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document from a room."""
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id, Document.project_id == project.id)
    )
    document = doc_result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    _safe_remove(document.file_path)
    await db.delete(document)
    await db.flush()
    logger.info("Room %d: deleted document id=%d (%r)", project.id, document_id, document.filename)


@router.post("/{room_id}/documents/{document_id}/process", response_model=ProcessDocumentResponse)
async def room_process_document(
    document_id: int,
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> ProcessDocumentResponse:
    """Full pipeline (embed + extract) for a document in a room."""
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id, Document.project_id == project.id)
    )
    document = doc_result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    embed_svc = OllamaEmbeddingService()
    try:
        embedded_chunks, total_chunks = await embed_svc.embed_document_chunks(document_id, db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding phase failed: {exc}")

    llm_svc = OllamaLLMService()
    try:
        summary = await llm_svc.process_document(document_id, db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction phase failed: {exc}")

    message = (
        f"Processed '{document.filename}': embedded {embedded_chunks}/{total_chunks} chunks, "
        f"saved {summary.concepts_saved} concepts and {summary.claims_saved} claims."
    )
    return ProcessDocumentResponse(
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
        message=message,
    )


@router.post("/{room_id}/documents/embed-all", response_model=EmbedAllResponse)
async def room_embed_all_documents(
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> EmbedAllResponse:
    """Embed all un-embedded chunks in this room."""
    docs_result = await db.execute(
        select(Document).where(Document.project_id == project.id)
    )
    documents = docs_result.scalars().all()

    if not documents:
        return EmbedAllResponse(documents_processed=0, total_embedded=0, total_chunks=0, message="No documents found.")

    svc = OllamaEmbeddingService()
    docs_processed = grand_embedded = grand_total = 0

    for doc in documents:
        async with AsyncSessionLocal() as doc_session:
            try:
                embedded, total = await svc.embed_document_chunks(doc.id, doc_session)
                grand_embedded += embedded
                grand_total += total
                docs_processed += 1
            except Exception as exc:
                logger.error("Room %d embed_all: failed doc id=%d: %s", project.id, doc.id, exc)
                await doc_session.rollback()

    return EmbedAllResponse(
        documents_processed=docs_processed,
        total_embedded=grand_embedded,
        total_chunks=grand_total,
        message=f"Embedded {grand_embedded} of {grand_total} chunks across {docs_processed} document(s).",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ROOM-SCOPED CONCEPT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/{room_id}/concepts/map", response_model=ReactFlowConceptMapResponse)
async def room_get_concept_map(
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> ReactFlowConceptMapResponse:
    """React Flow concept map for a specific room."""
    concept_res = await db.execute(
        select(Concept).where(Concept.project_id == project.id)
    )
    concepts: List[Concept] = list(concept_res.scalars().all())

    concept_ids_in_project = {c.id for c in concepts}
    rel_res = await db.execute(
        select(Relationship).where(Relationship.source_concept_id.in_(concept_ids_in_project))
    )
    relationships = list(rel_res.scalars().all())

    doc_counts: Dict[int, int] = {}
    if concept_ids_in_project:
        dc_res = await db.execute(
            select(Claim.concept_id, func.count(distinct(Claim.document_id)).label("doc_count"))
            .where(Claim.concept_id.in_(concept_ids_in_project))
            .group_by(Claim.concept_id)
        )
        doc_counts = {row.concept_id: row.doc_count for row in dc_res}

    brain_res = await db.execute(
        select(BrainState)
        .where(BrainState.project_id == project.id)
        .order_by(BrainState.last_updated.desc())
        .limit(1)
    )
    brain_state = brain_res.scalar_one_or_none()

    children_map: Dict[int, List[int]] = {}
    for c in concepts:
        if c.parent_concept_id is not None:
            children_map.setdefault(c.parent_concept_id, []).append(c.id)

    nodes: List[ReactFlowNode] = []
    gap_items: List[GapItem] = []
    _imp = {"critical": 0, "important": 1, "nice_to_have": 2}

    for c in concepts:
        node_type = "gap" if c.is_gap else "concept"
        cluster_id = f"cluster_{c.cluster_label}" if c.cluster_label is not None else None
        parent_id = f"concept_{c.parent_concept_id}" if c.parent_concept_id else None
        children = [f"concept_{cid}" for cid in children_map.get(c.id, [])]

        nodes.append(ReactFlowNode(
            id=f"concept_{c.id}",
            type=node_type,
            data=ReactFlowNodeData(
                label=c.name,
                description=c.description,
                coverage_score=c.coverage_score,
                consensus_score=c.consensus_score,
                generality_score=c.generality_score,
                document_count=doc_counts.get(c.id, 0),
                is_gap=c.is_gap,
                gap_type=c.gap_type.value if c.gap_type else None,
                cluster_id=cluster_id,
                parent_id=parent_id,
                children=children,
            ),
            position=ReactFlowPosition(x=0.0, y=0.0),
        ))

        if c.is_gap:
            meta = c.metadata_json or {}
            gap_items.append(GapItem(
                id=c.id,
                name=c.name,
                description=c.description,
                gap_type=GapTypeSchema(c.gap_type.value if c.gap_type else "missing_link"),
                gap_subtype=meta.get("gap_subtype", "weak_coverage"),
                importance=meta.get("importance"),
                suggestions=meta.get("suggestions", []),
                related_to=meta.get("related_to", []),
                is_synthetic=bool(meta.get("is_synthetic_gap", False)),
                coverage_score=c.coverage_score,
                generality_score=c.generality_score,
                cluster_label=c.cluster_label,
            ))

    gap_items.sort(key=lambda g: (_imp.get(g.importance or "important", 1), g.gap_subtype))

    edges: List[ReactFlowEdge] = []
    for r in relationships:
        if r.source_concept_id not in concept_ids_in_project or r.target_concept_id not in concept_ids_in_project:
            continue
        rel_type = r.relationship_type.value if r.relationship_type else "similar"
        edges.append(ReactFlowEdge(
            id=f"rel_{r.id}",
            source=f"concept_{r.source_concept_id}",
            target=f"concept_{r.target_concept_id}",
            type=rel_type,
            data=ReactFlowEdgeData(strength=r.strength, confidence=r.confidence, label=rel_type),
        ))

    cluster_ids = {c.cluster_label for c in concepts if c.cluster_label is not None}
    metadata = ReactFlowMapMetadata(
        total_concepts=len(concepts),
        total_relationships=len(edges),
        total_gaps=len(gap_items),
        num_clusters=len(cluster_ids),
        brain_last_updated=brain_state.last_updated if brain_state else None,
        has_clustering=bool(cluster_ids),
    )

    return ReactFlowConceptMapResponse(nodes=nodes, edges=edges, gaps=gap_items, metadata=metadata)


@router.post("/{room_id}/concepts/build", response_model=KnowledgeBuildResponse)
async def room_build_knowledge(
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeBuildResponse:
    """Rebuild the knowledge graph for a specific room."""
    pipeline = LacunaPipeline()
    t0 = time.monotonic()
    try:
        result = await pipeline.rebuild_project_knowledge(project.id, db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Knowledge build failed: {exc}")

    elapsed = round(time.monotonic() - t0, 2)
    return KnowledgeBuildResponse(
        project_id=result.project_id,
        concepts_before_normalize=result.concepts_before_normalize,
        concepts_after_normalize=result.concepts_after_normalize,
        merged_count=result.merged_count,
        num_clusters=result.num_clusters,
        algorithm_used=result.algorithm_used,
        clustered_concepts=result.clustered_concepts,
        relationships_found=result.relationships_found,
        relationships_saved=result.relationships_saved,
        expected_gaps=result.expected_gaps,
        bridging_gaps=result.bridging_gaps,
        weak_coverage_gaps=result.weak_coverage_gaps,
        total_gaps=result.total_gaps,
        concepts_scored=result.concepts_scored,
        strong_consensus_count=result.strong_consensus_count,
        contested_count=result.contested_count,
        contradiction_count=result.contradiction_count,
        brain_summary=result.brain_summary,
        processing_time_seconds=elapsed,
        message=result.message,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ROOM-SCOPED BRAIN ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/{room_id}/brain/chat", response_model=BrainChatResponse)
async def room_chat(
    request: BrainChatRequest,
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> BrainChatResponse:
    """RAG chat scoped to a specific room's documents."""
    try:
        brain_service = BrainService()
        result = await brain_service.chat(request.question, project.id, db, top_k=request.top_k)
        return BrainChatResponse(
            question=result.question,
            answer=result.answer,
            sources=result.sources,
            relevant_concepts=result.relevant_concepts,
            confidence=result.confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")


@router.get("/{room_id}/brain/status", response_model=BrainStatusResponse)
async def room_brain_status(
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> BrainStatusResponse:
    """Brain status for a specific room."""
    try:
        brain_res = await db.execute(
            select(BrainState).where(BrainState.project_id == project.id)
            .order_by(BrainState.last_updated.desc()).limit(1)
        )
        brain_state: Optional[BrainState] = brain_res.scalar_one_or_none()

        doc_count = (await db.execute(
            select(func.count()).select_from(Document).where(Document.project_id == project.id)
        )).scalar() or 0

        concept_count = (await db.execute(
            select(func.count()).select_from(Concept)
            .where(Concept.project_id == project.id, Concept.is_gap == False)
        )).scalar() or 0

        gap_count = (await db.execute(
            select(func.count()).select_from(Concept)
            .where(Concept.project_id == project.id, Concept.is_gap == True)
        )).scalar() or 0

        rel_count = (await db.execute(
            select(func.count()).select_from(Relationship)
            .join(Concept, Relationship.source_concept_id == Concept.id)
            .where(Concept.project_id == project.id)
        )).scalar() or 0

        avg_consensus = (await db.execute(
            select(func.avg(Concept.consensus_score))
            .where(Concept.project_id == project.id, Concept.is_gap == False, Concept.consensus_score.isnot(None))
        )).scalar()

        consensus_comp = float(avg_consensus) if avg_consensus is not None else 0.5
        health_score = round(
            0.40 * consensus_comp + 0.30 * min(doc_count / 10, 1.0) + 0.30 * min(concept_count / 50, 1.0), 4
        )

        return BrainStatusResponse(
            project_id=project.id,
            last_updated=brain_state.last_updated if brain_state else None,
            doc_count=doc_count,
            concept_count=concept_count,
            gap_count=gap_count,
            relationship_count=rel_count,
            avg_consensus=round(float(avg_consensus), 4) if avg_consensus is not None else None,
            health_score=health_score,
            summary_text=brain_state.summary_text if brain_state else None,
            has_brain=brain_state is not None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Status check failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# ROOM-SCOPED PIPELINE ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/{room_id}/pipeline/process-all", response_model=PipelineProcessAllResponse)
async def room_process_all(
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> PipelineProcessAllResponse:
    """Batch process all documents in a room and rebuild the knowledge graph."""
    docs_result = await db.execute(
        select(Document).where(Document.project_id == project.id)
    )
    documents = docs_result.scalars().all()

    if not documents:
        return PipelineProcessAllResponse(
            project_id=project.id, documents_processed=0, documents_failed=0,
            total_concepts=0, total_relationships=0, total_gaps=0,
            processing_time_seconds=0.0, errors=[], message="No documents found.",
        )

    pipeline = LacunaPipeline()
    t0 = time.monotonic()
    docs_processed = docs_failed = 0
    errors: List[str] = []

    for doc in documents:
        async with AsyncSessionLocal() as doc_session:
            try:
                await pipeline.process_document(doc.id, doc_session)
                docs_processed += 1
            except Exception as exc:
                docs_failed += 1
                errors.append(f"doc id={doc.id} ({doc.filename}): {str(exc)[:120]}")
                await doc_session.rollback()

    try:
        knowledge = await pipeline.rebuild_project_knowledge(project.id, db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Knowledge rebuild failed: {exc}")

    elapsed = round(time.monotonic() - t0, 2)
    message = (
        f"Processed {docs_processed}/{len(documents)} document(s) ({docs_failed} failed) in {elapsed}s. "
        f"Knowledge graph: {knowledge.num_clusters} cluster(s), {knowledge.relationships_saved} relationship(s), {knowledge.total_gaps} gap(s)."
    )

    return PipelineProcessAllResponse(
        project_id=project.id,
        documents_processed=docs_processed,
        documents_failed=docs_failed,
        total_concepts=knowledge.concepts_after_normalize,
        total_relationships=knowledge.relationships_saved,
        total_gaps=knowledge.total_gaps,
        processing_time_seconds=elapsed,
        errors=errors,
        message=message,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ROOM-SCOPED PHASED PIPELINE (BACKGROUND)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/{room_id}/pipeline/start", response_model=PipelineStartResponse)
async def room_pipeline_start(
    body: Optional[PipelineStartRequest] = None,
    project: Project = Depends(get_authorized_project),
    db: AsyncSession = Depends(get_db),
) -> PipelineStartResponse:
    """
    Launch the phased pipeline as a background task for this room.

    Optionally pass ``document_ids`` in the request body to only embed/extract
    specific (newly uploaded) documents.  The knowledge rebuild always runs
    across all project data.

    Returns immediately. Poll ``GET .../pipeline/status`` for progress.
    """
    from app.services.pipeline_manager import pipeline_manager

    if pipeline_manager.is_running(project.id):
        existing = pipeline_manager.get_status(project.id)
        return PipelineStartResponse(
            status="already_running",
            phase=existing.phase.value if existing else "unknown",
            total_documents=existing.total_documents if existing else 0,
        )

    doc_ids = body.document_ids if body else None

    # Count documents to process
    if doc_ids:
        doc_count = len(doc_ids)
    else:
        doc_count_result = await db.execute(
            select(func.count(Document.id)).where(Document.project_id == project.id)
        )
        doc_count = doc_count_result.scalar() or 0

    from app.services.pipeline_manager import PipelineStatus

    # Pre-create the status so we can pass it into the coroutine AND to start()
    # (get_status() would return None here because start() hasn't been called yet)
    ps = PipelineStatus(project_id=project.id, total_documents=doc_count)

    pipeline = LacunaPipeline()
    pipeline_manager.start(
        project.id,
        pipeline.process_all_phased(
            project.id,
            ps,
            document_ids=doc_ids,
        ),
        status=ps,
    )

    logger.info(
        "Room %d: phased pipeline started (%d documents%s)",
        project.id, doc_count,
        f", ids={doc_ids}" if doc_ids else " — all",
    )

    return PipelineStartResponse(
        status="started",
        phase=ps.phase.value,
        total_documents=doc_count,
    )


@router.get("/{room_id}/pipeline/status", response_model=PipelineStatusResponse)
async def room_pipeline_status(
    project: Project = Depends(get_authorized_project),
) -> PipelineStatusResponse:
    """Poll the current phased pipeline status for this room."""
    from app.services.pipeline_manager import pipeline_manager

    status = pipeline_manager.get_status(project.id)
    if status is None:
        return PipelineStatusResponse(phase="idle")

    return PipelineStatusResponse(
        phase=status.phase.value,
        total_documents=status.total_documents,
        documents_embedded=status.documents_embedded,
        documents_extracted=status.documents_extracted,
        documents_failed=status.documents_failed,
        current_document=status.current_document,
        errors=status.errors,
        elapsed_seconds=status.elapsed_seconds,
    )
