"""
Brain service: consensus scoring, brain state persistence, and RAG chat.

Public API
----------
BrainService.build_brain(project_id, db, *, clear_existing=False) -> BrainBuildResult
BrainService.chat(question, project_id, db, top_k=5)              -> ChatResult
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Set

import httpx
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database_models import (
    BrainState,
    Claim,
    ClaimType,
    Concept,
    Document,
    Relationship,
)
from app.services.embedding import OllamaEmbeddingService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConsensusSummary:
    """Per-project consensus breakdown produced by _score_all_concepts."""

    strong: List[str]        # concept names, score > 0.8
    contested: List[str]     # concept names, 0.3 <= score <= 0.8
    contradicted: List[str]  # concept names, score < 0.3
    avg_score: float


@dataclasses.dataclass
class BrainBuildResult:
    """Returned by BrainService.build_brain."""

    project_id: int
    concepts_scored: int
    strong_consensus_count: int
    contested_count: int
    contradiction_count: int
    summary_text: str
    message: str


@dataclasses.dataclass
class ChatResult:
    """Returned by BrainService.chat."""

    question: str
    answer: str
    sources: List[str]            # document filenames
    relevant_concepts: List[str]  # concept names
    confidence: float


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BRAIN_SUMMARY_PROMPT = """\
You are a research synthesis expert. Below is a structured summary of \
a research knowledge base.

## Concept Consensus Summary
- Strong consensus (score > 0.8): {strong}
- Contested areas (score 0.3–0.8): {contested}
- Key contradictions (score < 0.3): {contradicted}

## Statistics
- Total concepts: {total_concepts}
- Total relationships: {total_relationships}
- Total knowledge gaps detected: {total_gaps}

Write a concise 2-3 paragraph synthesis describing:
1. What this research collection covers and its areas of strong agreement.
2. What remains contested or under-explored.
3. The most important knowledge gaps that future research should address.

Be direct and specific. Do not invent facts not implied by the data above.

Synthesis:"""


_CHAT_PROMPT = """\
You are a knowledgeable research assistant with access to a curated knowledge base.

## User Question
{question}

## Relevant Text Excerpts
{chunks}

## Relevant Concepts
{concepts}

## Supporting Claims
{claims}

## Known Knowledge Gaps
{gaps}

Answer the user's question based strictly on the provided context. \
If the context is insufficient, say so and point to what additional \
research might be needed. Cite sources by filename where possible.

Answer:"""


# ---------------------------------------------------------------------------
# BrainService
# ---------------------------------------------------------------------------

class BrainService:
    """Consensus scoring, brain state persistence, and RAG chat."""

    # Claim-type weights used in the consensus scoring formula
    _SUPPORT_WEIGHTS: Dict[ClaimType, float] = {
        ClaimType.SUPPORTS: 1.0,
        ClaimType.EXTENDS: 0.7,
        ClaimType.COMPLEMENTS: 0.5,
    }
    _CONTRADICT_WEIGHTS: Dict[ClaimType, float] = {
        ClaimType.CONTRADICTS: 1.0,
    }

    def __init__(self) -> None:
        self._embedder = OllamaEmbeddingService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_brain(
        self,
        project_id: int,
        db: AsyncSession,
        *,
        clear_existing: bool = False,
    ) -> BrainBuildResult:
        """
        Full brain-building pipeline:

        1. Optionally clear an existing BrainState row.
        2. Compute & persist consensus_score for every concept.
        3. Gather project statistics.
        4. Generate an LLM synthesis summary.
        5. Upsert the BrainState row.
        6. Commit all changes.
        """
        if clear_existing:
            await self._clear_brain_state(project_id, db)

        # Step 2 — consensus scoring (writes concept.consensus_score)
        summary = await self._score_all_concepts(project_id, db)

        # Step 3 — project statistics
        total_concepts = await self._count_concepts(project_id, db)
        rel_count = await self._count_relationships(project_id, db)
        gap_count = await self._count_gaps(project_id, db)

        # Step 4 — LLM summary
        summary_text = await self._generate_summary(
            summary, total_concepts, rel_count, gap_count
        )

        # Step 5 — persist BrainState
        await self._persist_brain_state(project_id, db, summary_text, summary)

        # Step 6 — single commit for both concept scores + brain state
        await db.commit()

        return BrainBuildResult(
            project_id=project_id,
            concepts_scored=total_concepts,
            strong_consensus_count=len(summary.strong),
            contested_count=len(summary.contested),
            contradiction_count=len(summary.contradicted),
            summary_text=summary_text,
            message=(
                f"Brain built: {total_concepts} concepts scored, "
                f"{len(summary.strong)} strong consensus, "
                f"{len(summary.contested)} contested, "
                f"{len(summary.contradicted)} contradictions."
            ),
        )

    async def chat(
        self,
        question: str,
        project_id: int,
        db: AsyncSession,
        top_k: int = 5,
    ) -> ChatResult:
        """
        RAG pipeline:

        1. Embed the question.
        2. pgvector chunk similarity search (filtered to this project).
        3. pgvector concept similarity search.
        4. Gather claims + gap nodes for the matched concepts.
        5. Format context → LLM prompt → answer.
        """
        # 1 — embed
        q_emb = await self._embedder.embed_text(question)
        if q_emb is None:
            return ChatResult(
                question=question,
                answer="Unable to embed the question (Ollama may be unavailable).",
                sources=[],
                relevant_concepts=[],
                confidence=0.0,
            )

        # 2 — chunk search (fetch extra, then filter by project)
        project_doc_ids = await self._get_project_doc_ids(project_id, db)
        all_chunks = await self._embedder.find_similar_chunks(
            q_emb, db, top_k=top_k * 3, threshold=0.35
        )
        similar_chunks = [
            c for c in all_chunks if c["document_id"] in project_doc_ids
        ][:top_k]

        # 3 — relevant concepts via pgvector
        relevant_concepts = await self._find_relevant_concepts(
            q_emb, project_id, db, top_k=top_k
        )

        # 4 — claims + gaps
        concept_ids = [c["id"] for c in relevant_concepts]
        claims = await self._load_claims_for(concept_ids, db)
        gaps = await self._load_gaps_for(project_id, db)

        # 5 — build prompt + LLM call
        prompt = _CHAT_PROMPT.format(
            question=question,
            chunks=self._format_chunks(similar_chunks) or "(none)",
            concepts=self._format_concepts(relevant_concepts) or "(none)",
            claims=self._format_claims(claims) or "(none)",
            gaps=self._format_gaps(gaps),
        )
        answer = await self._call_llm(prompt, max_tokens=600)

        confidence = (
            sum(c["similarity"] for c in similar_chunks) / len(similar_chunks)
            if similar_chunks
            else 0.3
        )
        sources = list({c["filename"] for c in similar_chunks})
        concept_names = [c["name"] for c in relevant_concepts]

        return ChatResult(
            question=question,
            answer=answer or "No answer generated.",
            sources=sources,
            relevant_concepts=concept_names,
            confidence=min(confidence, 1.0),
        )

    # ------------------------------------------------------------------
    # Consensus scoring
    # ------------------------------------------------------------------

    async def _score_all_concepts(
        self, project_id: int, db: AsyncSession
    ) -> ConsensusSummary:
        """Compute and persist consensus_score for every concept in the project."""
        concept_result = await db.execute(
            select(Concept).where(Concept.project_id == project_id)
        )
        concepts = concept_result.scalars().all()

        if not concepts:
            return ConsensusSummary(
                strong=[], contested=[], contradicted=[], avg_score=0.5
            )

        # Load all claims for this project's concepts in one query
        concept_ids = [c.id for c in concepts]
        claim_result = await db.execute(
            select(Claim).where(Claim.concept_id.in_(concept_ids))
        )
        all_claims = claim_result.scalars().all()

        # Group by concept_id
        claims_by_concept: Dict[int, List[Claim]] = {}
        for claim in all_claims:
            claims_by_concept.setdefault(claim.concept_id, []).append(claim)

        strong: List[str] = []
        contested: List[str] = []
        contradicted: List[str] = []
        total_score = 0.0

        for concept in concepts:
            score = self._compute_consensus(claims_by_concept.get(concept.id, []))
            concept.consensus_score = score
            total_score += score
            if score > 0.8:
                strong.append(concept.name)
            elif score < 0.3:
                contradicted.append(concept.name)
            else:
                contested.append(concept.name)

        avg_score = total_score / len(concepts)
        return ConsensusSummary(
            strong=strong,
            contested=contested,
            contradicted=contradicted,
            avg_score=avg_score,
        )

    def _compute_consensus(self, claims: List[Claim]) -> float:
        """
        consensus_score = support_weight / (support_weight + contradict_weight)

        Weights:
          supports    → 1.0   (strong positive evidence)
          extends     → 0.7   (builds on, partial support)
          complements → 0.5   (neutral-positive)
          contradicts → 1.0   (strong negative evidence)

        Falls back to 0.5 (neutral) when there are no claims.
        """
        if not claims:
            return 0.5

        support = sum(self._SUPPORT_WEIGHTS.get(c.claim_type, 0.0) for c in claims)
        contra = sum(self._CONTRADICT_WEIGHTS.get(c.claim_type, 0.0) for c in claims)
        total = support + contra
        return support / total if total > 0 else 0.5

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _generate_summary(
        self,
        summary: ConsensusSummary,
        total_concepts: int,
        total_relationships: int,
        total_gaps: int,
    ) -> str:
        prompt = _BRAIN_SUMMARY_PROMPT.format(
            strong=", ".join(summary.strong[:10]) or "none",
            contested=", ".join(summary.contested[:10]) or "none",
            contradicted=", ".join(summary.contradicted[:10]) or "none",
            total_concepts=total_concepts,
            total_relationships=total_relationships,
            total_gaps=total_gaps,
        )
        raw = await self._call_llm(prompt, max_tokens=500)
        return raw.strip() if raw else "Insufficient data to generate synthesis."

    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Dispatch to Groq (if GROQ_API is set) or local Ollama."""
        if settings.GROQ_API:
            return await self._call_groq(prompt, max_tokens)
        return await self._call_ollama(prompt, max_tokens)

    async def _call_groq(self, prompt: str, max_tokens: int = 500) -> str:
        """POST to Groq chat completions (OpenAI-compatible) and return the text."""
        try:
            timeout = httpx.Timeout(float(settings.GROQ_TIMEOUT), connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.GROQ_API}"},
                    json={
                        "model": settings.GROQ_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.3,
                    },
                )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logger.error("_call_groq: HTTP %d: %s", resp.status_code, resp.text[:200])
            return ""
        except Exception as exc:
            logger.error("_call_groq error: %s", exc)
            return ""

    async def _call_ollama(self, prompt: str, max_tokens: int = 500) -> str:
        """POST to local Ollama /api/generate and return the response text."""
        payload = {
            "model": settings.OLLAMA_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        timeout = httpx.Timeout(settings.OLLAMA_TIMEOUT, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                )
            if resp.status_code != 200:
                logger.error("_call_ollama returned %d: %s", resp.status_code, resp.text[:200])
                return ""
            return resp.json().get("response", "")
        except Exception as exc:
            logger.error("_call_ollama error: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _persist_brain_state(
        self,
        project_id: int,
        db: AsyncSession,
        summary_text: str,
        summary: ConsensusSummary,
    ) -> None:
        result = await db.execute(
            select(BrainState)
            .where(BrainState.project_id == project_id)
            .order_by(BrainState.last_updated.desc())
            .limit(1)
        )
        brain_state = result.scalar_one_or_none()

        consensus_json = {
            "strong_consensus": summary.strong[:20],
            "contested": summary.contested[:20],
            "contradictions": summary.contradicted[:20],
            "avg_score": round(summary.avg_score, 4),
        }

        if brain_state:
            brain_state.summary_text = summary_text
            brain_state.consensus_json = consensus_json
        else:
            brain_state = BrainState(
                project_id=project_id,
                summary_text=summary_text,
                consensus_json=consensus_json,
            )
            db.add(brain_state)

    async def _clear_brain_state(self, project_id: int, db: AsyncSession) -> None:
        result = await db.execute(
            select(BrainState).where(BrainState.project_id == project_id)
        )
        for state in result.scalars().all():
            await db.delete(state)
        await db.flush()

    async def _count_concepts(self, project_id: int, db: AsyncSession) -> int:
        result = await db.execute(
            select(func.count())
            .select_from(Concept)
            .where(Concept.project_id == project_id)
        )
        return result.scalar() or 0

    async def _count_gaps(self, project_id: int, db: AsyncSession) -> int:
        result = await db.execute(
            select(func.count())
            .select_from(Concept)
            .where(Concept.project_id == project_id, Concept.is_gap == True)
        )
        return result.scalar() or 0

    async def _count_relationships(self, project_id: int, db: AsyncSession) -> int:
        """Count relationships whose source concept belongs to the project."""
        result = await db.execute(
            select(func.count())
            .select_from(Relationship)
            .join(Concept, Relationship.source_concept_id == Concept.id)
            .where(Concept.project_id == project_id)
        )
        return result.scalar() or 0

    async def _get_project_doc_ids(
        self, project_id: int, db: AsyncSession
    ) -> Set[int]:
        result = await db.execute(
            select(Document.id).where(Document.project_id == project_id)
        )
        return {row[0] for row in result.all()}

    async def _find_relevant_concepts(
        self,
        q_emb: List[float],
        project_id: int,
        db: AsyncSession,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """pgvector cosine similarity search over non-gap project concepts."""
        embedding_str = "[" + ",".join(f"{v:.8f}" for v in q_emb) + "]"
        sql = text(
            """
            SELECT
                c.id,
                c.name,
                c.description,
                c.coverage_score,
                c.consensus_score,
                1 - (c.embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM concepts c
            WHERE c.project_id = :project_id
              AND c.embedding IS NOT NULL
              AND c.is_gap = false
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
            """
        )
        result = await db.execute(
            sql,
            {"embedding": embedding_str, "project_id": project_id, "top_k": top_k},
        )
        rows = result.mappings().all()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "coverage_score": row["coverage_score"],
                "consensus_score": row["consensus_score"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]

    async def _load_claims_for(
        self, concept_ids: List[int], db: AsyncSession
    ) -> List[Dict[str, Any]]:
        if not concept_ids:
            return []
        result = await db.execute(
            select(Claim).where(Claim.concept_id.in_(concept_ids))
        )
        claims = result.scalars().all()
        return [
            {
                "claim_text": c.claim_text,
                "claim_type": c.claim_type.value,
                "confidence": c.confidence,
                "concept_id": c.concept_id,
            }
            for c in claims
        ]

    async def _load_gaps_for(
        self, project_id: int, db: AsyncSession
    ) -> List[Dict[str, Any]]:
        result = await db.execute(
            select(Concept)
            .where(Concept.project_id == project_id, Concept.is_gap == True)
            .limit(5)
        )
        gaps = result.scalars().all()
        return [
            {
                "name": g.name,
                "description": g.description,
                "gap_type": g.gap_type.value if g.gap_type else None,
            }
            for g in gaps
        ]

    # ------------------------------------------------------------------
    # Prompt context formatters
    # ------------------------------------------------------------------

    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            snippet = c["content"][:400].replace("\n", " ")
            parts.append(f"[{i}] ({c['filename']}) {snippet}")
        return "\n".join(parts)

    def _format_concepts(self, concepts: List[Dict[str, Any]]) -> str:
        parts = []
        for c in concepts:
            desc = (c.get("description") or "")[:150]
            score = c.get("consensus_score")
            score_tag = f" [consensus: {score:.2f}]" if score is not None else ""
            parts.append(f"- {c['name']}{score_tag}: {desc}")
        return "\n".join(parts)

    def _format_claims(self, claims: List[Dict[str, Any]]) -> str:
        parts = []
        for c in claims[:12]:
            parts.append(f"- [{c['claim_type']}] {c['claim_text'][:200]}")
        return "\n".join(parts)

    def _format_gaps(self, gaps: List[Dict[str, Any]]) -> str:
        if not gaps:
            return "(no gaps detected)"
        parts = []
        for g in gaps:
            desc = (g.get("description") or "")[:120]
            parts.append(f"- {g['name']} ({g['gap_type']}): {desc}")
        return "\n".join(parts)
