"""
Relationship detection and scoring service for concept pairs.

RelationshipDetector implements a 5-step pipeline:
  1. Candidate pair generation via embedding cosine-similarity pre-filter
     (0.30 ≤ sim ≤ 0.85).  Below 0.30 = too unrelated; above 0.85 = should
     have been merged in normalisation.
  2. Evidence gathering — claims for each concept + chunks where BOTH concept
     names appear as substrings (co-occurrence).
  3. LLM-based relationship classification with structured JSON output.
  4. Multi-signal strength scoring:
       0.30 · embedding_similarity
     + 0.25 · normalised_co_occurrence
     + 0.25 · has_shared_document (binary)
     + 0.20 · llm_confidence
  5. Deduplication (bidirectional extends → complements; conflict resolution)
     and DB persistence.

LLM type → DB RelationshipType mapping:
  "supports"   → SIMILAR    (no "supports" in RelationshipType enum)
  "extends"    → BUILDS_ON
  "contradicts"→ CONTRADICTS
  "complements"→ COMPLEMENTS
  "none"       → (skip — no meaningful relationship)

Backward-compatible RelationshipService class is preserved at the bottom
for any existing callers.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database_models import (
    Chunk,
    Claim,
    Concept,
    Document,
    Relationship,
    RelationshipType,
)
from app.services.llm_extractor import OllamaLLMService
from app.utils.helpers import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RelationshipResult:
    """Returned by RelationshipDetector.detect_relationships()."""

    project_id: int
    total_pairs_considered: int
    relationships_found: int
    relationships_saved: int
    by_type_count: Dict[str, int]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
You are analyzing the relationship between two research concepts based on \
evidence from academic documents.

Concept A: {concept_a_name} — {concept_a_description}
Concept B: {concept_b_name} — {concept_b_description}

Evidence from documents where both concepts appear:
---
{evidence_chunks}
---

Claims about Concept A: {claims_a}
Claims about Concept B: {claims_b}

What is the relationship between these concepts? Choose ONE:
- "supports": B provides evidence for or validates A (or vice versa)
- "contradicts": B challenges, refutes, or conflicts with A (or vice versa)
- "extends": B builds upon or advances A (directional — specify which extends which)
- "complements": A and B cover different aspects of the same broader topic
- "none": No meaningful relationship despite appearing together

Respond ONLY in valid JSON:
{{"relationship_type": "...", "direction": "a_to_b or b_to_a or bidirectional", "reasoning": "one sentence explanation", "confidence": 0.0}}\
"""

_CLASSIFY_RETRY_PROMPT = """\
Classify the relationship between two research concepts.

Concept A: {concept_a_name}
Concept B: {concept_b_name}

Choose ONE relationship type: supports, contradicts, extends, complements, none.

Return ONLY valid JSON — nothing else:
{{"relationship_type": "complements", "direction": "bidirectional", "reasoning": "Both relate to the same research area.", "confidence": 0.5}}\
"""

# ---------------------------------------------------------------------------
# LLM type → DB enum mapping
# ---------------------------------------------------------------------------

_VALID_LLM_TYPES: frozenset = frozenset(
    {"supports", "contradicts", "extends", "complements", "none"}
)

# "supports" → SIMILAR  (DB has no "supports" relationship type)
# "extends"  → BUILDS_ON
_LLM_TO_DB: Dict[str, Optional[RelationshipType]] = {
    "supports":    RelationshipType.SIMILAR,
    "extends":     RelationshipType.BUILDS_ON,
    "contradicts": RelationshipType.CONTRADICTS,
    "complements": RelationshipType.COMPLEMENTS,
    "none":        None,
}


# ---------------------------------------------------------------------------
# RelationshipDetector
# ---------------------------------------------------------------------------

class RelationshipDetector:
    """
    Detects semantic relationships between all concept pairs in a project.

    Designed for projects with 50–200 concepts.  LLM calls are bounded by
    MAX_LLM_CALLS (default 500) — the highest-similarity candidate pairs are
    processed first so the most informative edges are always captured.
    """

    # ---- tunables ----------------------------------------------------------
    MAX_LLM_CALLS: int = 500       # hard cap on LLM calls per run
    SIM_LOW: float = 0.30          # pairs below → too unrelated to analyse
    SIM_HIGH: float = 0.85         # pairs above → should have been merged
    MAX_EVIDENCE_CHUNKS: int = 3   # max co-occurring chunks sent to LLM
    MAX_CLAIMS_PER_CONCEPT: int = 4  # max claims per concept sent to LLM
    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        self._llm = OllamaLLMService()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def detect_relationships(
        self,
        project_id: int,
        db: AsyncSession,
    ) -> RelationshipResult:
        """Run the full 5-step detection pipeline and return a summary."""

        # ---- Step 1: load concepts with embeddings -----------------------
        logger.info(
            "RelationshipDetector: loading concepts for project %d", project_id
        )
        concepts = await self._load_concepts_with_embeddings(project_id, db)
        n = len(concepts)
        if n < 2:
            logger.warning(
                "RelationshipDetector: only %d concept(s) — nothing to do.", n
            )
            return RelationshipResult(project_id, 0, 0, 0, {})
        logger.info("RelationshipDetector: %d concepts loaded", n)

        # ---- Step 1b: load already-stored pairs (avoid redundant work) ---
        existing_pairs: Set[Tuple[int, int]] = await self._load_existing_pairs(db)
        logger.info(
            "RelationshipDetector: %d existing relationships already in DB",
            len(existing_pairs),
        )

        # ---- Step 2: candidate pair generation ---------------------------
        candidate_pairs = self._build_candidate_pairs(concepts)
        logger.info(
            "RelationshipDetector: %d raw candidate pairs (%.2f ≤ sim ≤ %.2f)",
            len(candidate_pairs),
            self.SIM_LOW,
            self.SIM_HIGH,
        )

        # Remove pairs that already exist in the DB
        fresh_pairs: List[Tuple[int, int, float]] = [
            (ci, cj, sim)
            for ci, cj, sim in candidate_pairs
            if (concepts[ci].id, concepts[cj].id) not in existing_pairs
            and (concepts[cj].id, concepts[ci].id) not in existing_pairs
        ]
        total_pairs_considered = len(fresh_pairs)
        logger.info(
            "RelationshipDetector: %d fresh pairs after removing %d already-stored",
            total_pairs_considered,
            len(candidate_pairs) - total_pairs_considered,
        )

        # Cap to MAX_LLM_CALLS — list is already sorted by descending similarity
        if len(fresh_pairs) > self.MAX_LLM_CALLS:
            fresh_pairs = fresh_pairs[: self.MAX_LLM_CALLS]
            logger.info(
                "RelationshipDetector: capped to %d pairs (MAX_LLM_CALLS=%d)",
                self.MAX_LLM_CALLS,
                self.MAX_LLM_CALLS,
            )

        if not fresh_pairs:
            logger.info("RelationshipDetector: no new pairs to process.")
            return RelationshipResult(project_id, 0, 0, 0, {})

        # ---- Step 3a: pre-load evidence ----------------------------------
        logger.info(
            "RelationshipDetector: loading evidence (claims + chunk mentions)..."
        )
        claims_by_concept = await self._load_claims_by_concept(project_id, db)
        chunks_by_id, chunk_mentions = await self._load_chunk_evidence(
            project_id, concepts, db
        )

        # Max co-occurrence count across all concepts (used to normalise the
        # co-occurrence component of the strength score).
        max_co_occ: int = max(
            (len(v) for v in chunk_mentions.values()), default=1
        ) or 1

        # ---- Steps 4 + 5: LLM classify + strength scoring ----------------
        detected: List[Dict[str, Any]] = []
        total = len(fresh_pairs)

        for idx, (ci, cj, sim) in enumerate(fresh_pairs):
            ca = concepts[ci]
            cb = concepts[cj]

            if idx % 50 == 0 or idx == total - 1:
                logger.info(
                    "RelationshipDetector: pair %d/%d — '%s' ↔ '%s' (sim=%.3f)",
                    idx + 1,
                    total,
                    ca.name,
                    cb.name,
                    sim,
                )

            evidence = self._gather_evidence(
                ca, cb, claims_by_concept, chunk_mentions, chunks_by_id
            )

            llm_result = await self._llm_classify(ca, cb, evidence)
            if llm_result is None:
                continue

            rel_type_str = llm_result["relationship_type"]
            if rel_type_str == "none":
                continue

            db_type: Optional[RelationshipType] = _LLM_TO_DB.get(rel_type_str)
            if db_type is None:
                continue

            # Resolve direction → source/target IDs
            direction = llm_result.get("direction", "bidirectional")
            if direction == "b_to_a":
                source_id, target_id = cb.id, ca.id
            else:
                # a_to_b or bidirectional: canonical ordering = lower id first
                source_id, target_id = (
                    (ca.id, cb.id) if ca.id < cb.id else (cb.id, ca.id)
                )

            # "extends" + bidirectional → treat as "complements" (no clear direction)
            if rel_type_str == "extends" and direction == "bidirectional":
                db_type = RelationshipType.COMPLEMENTS
                rel_type_str = "complements"

            co_count = len(evidence["co_occurring_ids"])
            shared = 1 if evidence["shared_doc_count"] > 0 else 0
            confidence = llm_result["confidence"]

            strength = self._compute_strength(
                sim, co_count, max_co_occ, shared, confidence
            )

            detected.append(
                {
                    "source_concept_id": source_id,
                    "target_concept_id": target_id,
                    "relationship_type": db_type,
                    "relationship_type_str": rel_type_str,
                    "direction": direction,
                    "strength": strength,
                    "confidence": confidence,
                    "evidence_json": {
                        "reasoning": llm_result.get("reasoning", ""),
                        "embedding_similarity": round(sim, 4),
                        "co_occurrence_chunks": co_count,
                        "evidence_chunk_previews": [
                            t[:200] for t in evidence["chunk_texts"][:2]
                        ],
                    },
                }
            )

        logger.info(
            "RelationshipDetector: LLM produced %d candidate relationships "
            "from %d pair(s) processed",
            len(detected),
            total,
        )

        # ---- Step 5: deduplication ---------------------------------------
        deduped = self._deduplicate(detected)
        logger.info(
            "RelationshipDetector: %d relationships after deduplication (was %d)",
            len(deduped),
            len(detected),
        )

        # ---- Step 6: persist to DB ---------------------------------------
        saved = await self._save_relationships(deduped, existing_pairs, db)
        logger.info(
            "RelationshipDetector: %d new relationships saved to DB", saved
        )

        # ---- summary -----------------------------------------------------
        by_type: Dict[str, int] = {}
        for r in deduped:
            t = r["relationship_type_str"]
            by_type[t] = by_type.get(t, 0) + 1

        return RelationshipResult(
            project_id=project_id,
            total_pairs_considered=total_pairs_considered,
            relationships_found=len(deduped),
            relationships_saved=saved,
            by_type_count=by_type,
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    async def _load_concepts_with_embeddings(
        self,
        project_id: int,
        db: AsyncSession,
    ) -> List[Concept]:
        result = await db.execute(
            select(Concept).where(
                Concept.project_id == project_id,
                Concept.embedding.isnot(None),
            )
        )
        return list(result.scalars().all())

    async def _load_existing_pairs(
        self,
        db: AsyncSession,
    ) -> Set[Tuple[int, int]]:
        result = await db.execute(
            select(
                Relationship.source_concept_id,
                Relationship.target_concept_id,
            )
        )
        return {(row[0], row[1]) for row in result.fetchall()}

    async def _load_claims_by_concept(
        self,
        project_id: int,
        db: AsyncSession,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Return {concept_id: [{claim_text, claim_type, confidence, document_id}]}."""
        result = await db.execute(
            select(Claim)
            .join(Concept, Claim.concept_id == Concept.id)
            .where(Concept.project_id == project_id)
        )
        claims_map: Dict[int, List[Dict[str, Any]]] = {}
        for claim in result.scalars().all():
            claim_type_val = (
                claim.claim_type.value
                if hasattr(claim.claim_type, "value")
                else str(claim.claim_type)
            )
            claims_map.setdefault(claim.concept_id, []).append(
                {
                    "claim_text": claim.claim_text,
                    "claim_type": claim_type_val,
                    "confidence": claim.confidence,
                    "document_id": claim.document_id,
                }
            )
        return claims_map

    async def _load_chunk_evidence(
        self,
        project_id: int,
        concepts: List[Concept],
        db: AsyncSession,
    ) -> Tuple[Dict[int, str], Dict[int, Set[int]]]:
        """
        Load all project chunks and build a concept → chunk mention index via
        case-insensitive substring search.

        Returns:
            chunks_by_id    : chunk_id → content text
            chunk_mentions  : concept_id → set of chunk_ids where name appears
        """
        result = await db.execute(
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.project_id == project_id,
                Chunk.content.isnot(None),
            )
        )
        all_chunks = result.scalars().all()
        logger.info(
            "RelationshipDetector: loaded %d chunks for co-occurrence search",
            len(all_chunks),
        )

        chunks_by_id: Dict[int, str] = {c.id: c.content for c in all_chunks}

        # Pre-lowercase content once to avoid repeated .lower() in the inner loop
        chunks_lower: Dict[int, str] = {
            c.id: c.content.lower() for c in all_chunks
        }

        chunk_mentions: Dict[int, Set[int]] = {}
        for concept in concepts:
            name_lower = concept.name.lower()
            matching: Set[int] = {
                cid
                for cid, text_lower in chunks_lower.items()
                if name_lower in text_lower
            }
            chunk_mentions[concept.id] = matching

        return chunks_by_id, chunk_mentions

    # ------------------------------------------------------------------
    # Step 2: candidate pair generation
    # ------------------------------------------------------------------

    def _build_candidate_pairs(
        self,
        concepts: List[Concept],
    ) -> List[Tuple[int, int, float]]:
        """
        Compute pairwise cosine similarities with numpy and return upper-triangle
        pairs in the window [SIM_LOW, SIM_HIGH], sorted descending by similarity.
        """
        n = len(concepts)

        # Build embedding matrix  (n, dim)
        embs_raw: List[List[float]] = []
        for c in concepts:
            try:
                embs_raw.append(list(c.embedding))
            except (TypeError, ValueError):
                embs_raw.append([0.0] * 768)

        embs = np.array(embs_raw, dtype=np.float32)

        # L2-normalise rows (embeddings should already be unit-length, but
        # re-normalising is cheap and makes the dot product = cosine similarity)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        embs = embs / norms

        # Full cosine similarity matrix via matrix multiplication  (n, n)
        sim_matrix = (embs @ embs.T).astype(np.float32)

        # Upper-triangle mask (exclude diagonal and lower half)
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid = upper & (sim_matrix >= self.SIM_LOW) & (sim_matrix <= self.SIM_HIGH)

        i_idx, j_idx = np.where(valid)
        similarities = sim_matrix[i_idx, j_idx]

        # Sort descending: strongest signal first (also determines which pairs
        # survive the MAX_LLM_CALLS cap)
        order = np.argsort(-similarities)
        pairs: List[Tuple[int, int, float]] = [
            (int(i_idx[k]), int(j_idx[k]), float(similarities[k]))
            for k in order
        ]
        return pairs

    # ------------------------------------------------------------------
    # Step 3: evidence gathering
    # ------------------------------------------------------------------

    def _gather_evidence(
        self,
        ca: Concept,
        cb: Concept,
        claims_by_concept: Dict[int, List[Dict[str, Any]]],
        chunk_mentions: Dict[int, Set[int]],
        chunks_by_id: Dict[int, str],
    ) -> Dict[str, Any]:
        """Assemble claims and co-occurring chunk text for an (A, B) pair."""
        all_claims_a = claims_by_concept.get(ca.id, [])
        all_claims_b = claims_by_concept.get(cb.id, [])

        # Shared-document check: any document that has claims for BOTH concepts?
        docs_a = {c["document_id"] for c in all_claims_a}
        docs_b = {c["document_id"] for c in all_claims_b}
        shared_doc_count = len(docs_a & docs_b)

        # Limit to what we'll actually send to the LLM
        claims_a = all_claims_a[: self.MAX_CLAIMS_PER_CONCEPT]
        claims_b = all_claims_b[: self.MAX_CLAIMS_PER_CONCEPT]

        # Co-occurring chunks: chunks that mention BOTH concept names
        chunks_a = chunk_mentions.get(ca.id, set())
        chunks_b = chunk_mentions.get(cb.id, set())
        co_ids: Set[int] = chunks_a & chunks_b

        chunk_texts: List[str] = [
            chunks_by_id[cid]
            for cid in list(co_ids)[: self.MAX_EVIDENCE_CHUNKS]
            if cid in chunks_by_id
        ]

        return {
            "claims_a": claims_a,
            "claims_b": claims_b,
            "co_occurring_ids": co_ids,
            "chunk_texts": chunk_texts,
            "shared_doc_count": shared_doc_count,
        }

    # ------------------------------------------------------------------
    # Step 4: LLM classification
    # ------------------------------------------------------------------

    async def _llm_classify(
        self,
        ca: Concept,
        cb: Concept,
        evidence: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Call the LLM to classify the relationship type and direction.

        Returns a dict with keys: relationship_type, direction, reasoning,
        confidence — or None if the LLM call / JSON parse failed.
        """
        # Format co-occurring chunks
        if evidence["chunk_texts"]:
            chunks_formatted = [
                f"[Chunk {i + 1}]: {text[:400]}"
                for i, text in enumerate(evidence["chunk_texts"])
            ]
            evidence_chunks_str = "\n".join(chunks_formatted)
        else:
            evidence_chunks_str = (
                "(No direct co-occurrence found — "
                "classify based on concept descriptions only)"
            )

        # Format claims
        claims_a_str = (
            "; ".join(
                f"{c['claim_type']}: {c['claim_text'][:120]}"
                for c in evidence["claims_a"]
            )
            or "(none)"
        )
        claims_b_str = (
            "; ".join(
                f"{c['claim_type']}: {c['claim_text'][:120]}"
                for c in evidence["claims_b"]
            )
            or "(none)"
        )

        prompt = _CLASSIFY_PROMPT.format(
            concept_a_name=ca.name,
            concept_a_description=(ca.description or "No description available")[:200],
            concept_b_name=cb.name,
            concept_b_description=(cb.description or "No description available")[:200],
            evidence_chunks=evidence_chunks_str,
            claims_a=claims_a_str,
            claims_b=claims_b_str,
        )
        retry_prompt = _CLASSIFY_RETRY_PROMPT.format(
            concept_a_name=ca.name,
            concept_b_name=cb.name,
        )

        success, parsed = await self._llm._call_llm_json(
            prompt, retry_prompt=retry_prompt
        )
        if not success or not isinstance(parsed, dict):
            logger.warning(
                "RelationshipDetector: LLM parse failed for '%s' ↔ '%s'",
                ca.name,
                cb.name,
            )
            return None

        rel_type = str(parsed.get("relationship_type", "none")).lower().strip()
        if rel_type not in _VALID_LLM_TYPES:
            rel_type = "none"

        direction = str(parsed.get("direction", "bidirectional")).lower().strip()
        if direction not in {"a_to_b", "b_to_a", "bidirectional"}:
            direction = "bidirectional"

        raw_conf = parsed.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(raw_conf)))
        except (TypeError, ValueError):
            confidence = 0.5

        reasoning = str(parsed.get("reasoning", "")).strip()[:300]

        logger.debug(
            "RelationshipDetector: '%s' ↔ '%s' → %s [%s] conf=%.2f",
            ca.name,
            cb.name,
            rel_type,
            direction,
            confidence,
        )
        return {
            "relationship_type": rel_type,
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Step 4b: multi-signal strength scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_strength(
        sim: float,
        co_count: int,
        max_co_occ: int,
        shared_docs: int,
        confidence: float,
    ) -> float:
        """
        strength = 0.30 · embedding_similarity
                 + 0.25 · normalised_co_occurrence
                 + 0.25 · has_shared_document   (binary 0/1)
                 + 0.20 · llm_confidence
        """
        norm_co = co_count / max_co_occ if max_co_occ > 0 else 0.0
        s = (
            0.30 * sim
            + 0.25 * norm_co
            + 0.25 * (1.0 if shared_docs > 0 else 0.0)
            + 0.20 * confidence
        )
        return round(max(0.0, min(1.0, s)), 4)

    # ------------------------------------------------------------------
    # Step 5: deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(
        detected: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge or resolve duplicate/conflicting detections for the same concept pair.

        Rules:
        - If A→B "extends" *and* B→A "extends" were both found (opposite directions
          for the same pair), merge into a single bidirectional "complements".
        - If the same pair has conflicting types, keep the detection with higher
          confidence; record the discarded type in evidence_json.
        - If the same pair has the same type, keep the higher-confidence version.

        The canonical key is (min_id, max_id) so (A,B) and (B,A) collapse to one.
        """
        # best[canonical_key] = winning detection dict
        best: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for r in detected:
            src, tgt = r["source_concept_id"], r["target_concept_id"]
            key: Tuple[int, int] = (min(src, tgt), max(src, tgt))

            if key not in best:
                best[key] = r
                continue

            existing = best[key]

            # Both are "extends" in opposite directions → merge into "complements"
            if (
                r["relationship_type_str"] == "extends"
                and existing["relationship_type_str"] == "extends"
                and r.get("direction") != existing.get("direction")
            ):
                existing["relationship_type"] = RelationshipType.COMPLEMENTS
                existing["relationship_type_str"] = "complements"
                existing["direction"] = "bidirectional"
                existing["confidence"] = round(
                    (existing["confidence"] + r["confidence"]) / 2, 4
                )
                existing["strength"] = round(
                    (existing["strength"] + r["strength"]) / 2, 4
                )
                existing["evidence_json"][
                    "merged_bidirectional_extends"
                ] = True
                continue

            # Conflicting or same type: keep the higher-confidence detection
            if r["confidence"] > existing["confidence"]:
                r["evidence_json"]["replaced_type"] = existing[
                    "relationship_type_str"
                ]
                r["evidence_json"]["replaced_confidence"] = existing["confidence"]
                best[key] = r

        return list(best.values())

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    async def _save_relationships(
        self,
        relationships: List[Dict[str, Any]],
        existing_pairs: Set[Tuple[int, int]],
        db: AsyncSession,
    ) -> int:
        """Persist new Relationship rows; return count actually saved."""
        saved = 0
        for r in relationships:
            src = r["source_concept_id"]
            tgt = r["target_concept_id"]

            # Guard against duplicates introduced during this run
            if (src, tgt) in existing_pairs or (tgt, src) in existing_pairs:
                logger.debug(
                    "RelationshipDetector: skipping already-stored pair (%d, %d)",
                    src,
                    tgt,
                )
                continue

            db.add(
                Relationship(
                    source_concept_id=src,
                    target_concept_id=tgt,
                    relationship_type=r["relationship_type"],
                    strength=r["strength"],
                    confidence=r["confidence"],
                    evidence_json=r.get("evidence_json", {}),
                )
            )
            existing_pairs.add((src, tgt))
            saved += 1

        if saved:
            await db.commit()
        return saved


# ---------------------------------------------------------------------------
# Backward-compatible RelationshipService
# (kept so existing imports / callers continue to work)
# ---------------------------------------------------------------------------

class RelationshipService:
    """
    Legacy relationship service — kept for backward compatibility.

    New code should use RelationshipDetector which provides the full
    project-level detection pipeline backed by evidence and LLM classification.
    """

    def __init__(self) -> None:
        from app.services.llm_extractor import LLMExtractor
        self.llm_extractor = LLMExtractor()

    async def detect_relationships(
        self,
        concepts: List[Dict[str, Any]],
        claims: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        use_llm: bool = False,
    ) -> List[Dict[str, Any]]:
        relationships = []
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i + 1 :]:
                if concept1.get("embedding") and concept2.get("embedding"):
                    sim = cosine_similarity(
                        concept1["embedding"], concept2["embedding"]
                    )
                    if sim > 0.3:
                        rel = await self._analyze_relationship(
                            concept1, concept2, sim, claims, use_llm
                        )
                        if rel:
                            relationships.append(rel)
        logger.info("RelationshipService: detected %d relationships", len(relationships))
        return relationships

    async def _analyze_relationship(
        self,
        concept1: Dict[str, Any],
        concept2: Dict[str, Any],
        embedding_similarity: float,
        claims: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        use_llm: bool = False,
    ) -> Optional[Dict[str, Any]]:
        rel_type, strength = self._infer_type(
            concept1, concept2, embedding_similarity, claims
        )
        confidence = 0.7
        evidence: Dict[str, Any] = {}
        if use_llm:
            llm_result = await self.llm_extractor.analyze_relationships(
                concept1["name"],
                concept2["name"],
                context=(
                    f"{concept1.get('description', '')} "
                    f"{concept2.get('description', '')}"
                ),
            )
            if llm_result:
                rel_type = llm_result["relationship_type"]
                confidence = llm_result["confidence"]
                evidence["llm_explanation"] = llm_result.get("explanation", "")
        return {
            "source_concept_id": concept1["id"],
            "target_concept_id": concept2["id"],
            "relationship_type": rel_type,
            "strength": strength,
            "confidence": confidence,
            "evidence_json": evidence,
        }

    def _infer_type(
        self,
        concept1: Dict[str, Any],
        concept2: Dict[str, Any],
        similarity: float,
        claims: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> Tuple[str, float]:
        gen1 = concept1.get("generality_score", 0.5)
        gen2 = concept2.get("generality_score", 0.5)
        gen_diff = abs(gen1 - gen2)
        if gen_diff > 0.3 and similarity > 0.6:
            return "parent_child", similarity
        if gen1 > gen2 + 0.2:
            return "prerequisite", similarity * 0.8
        if gen_diff < 0.2 and similarity > 0.7:
            return "complements", similarity
        if 0.5 < similarity < 0.7 and gen2 > gen1:
            return "builds_on", similarity
        return "similar", similarity

    async def score_relationships(
        self,
        relationships: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        concept_map = {c["id"]: c for c in concepts}
        type_weights = {
            "prerequisite": 1.0,
            "contradicts": 0.95,
            "builds_on": 0.9,
            "parent_child": 0.85,
            "complements": 0.7,
            "similar": 0.5,
        }
        for rel in relationships:
            source = concept_map.get(rel["source_concept_id"])
            target = concept_map.get(rel["target_concept_id"])
            if not source or not target:
                continue
            coverage_score = (
                source.get("coverage_score", 0.5) + target.get("coverage_score", 0.5)
            ) / 2
            type_weight = type_weights.get(rel["relationship_type"], 0.5)
            rel["importance_score"] = (
                rel["strength"] * 0.4 + coverage_score * 0.3 + type_weight * 0.3
            )
        relationships.sort(
            key=lambda x: x.get("importance_score", 0), reverse=True
        )
        return relationships
