"""
Knowledge gap detection service — Lacuna's signature feature.

"Lacuna" literally means gap.  This service finds three distinct kinds of
knowledge gaps in a user's document collection:

Pass 1 — Expected Topics   (LLM-based)
    The LLM is given all existing concepts and asked which domain topics are
    conspicuously absent.  A new synthetic Concept row is created for each
    suggestion so the gap appears on the concept map.

Pass 2 — Bridging Gaps     (graph + LLM)
    Clusters that are semantically related (centroid cosine similarity ≥ 0.40)
    but have NO inter-cluster Relationship rows represent areas where the
    literature hasn't connected two related sub-fields.  The LLM names the
    bridging concept; its embedding is placed at the average of the two
    cluster centroids.

Pass 3 — Weak Coverage     (statistical, no LLM)
    Existing concepts that are broad (generality_score > 0.50) but barely
    covered (coverage_score < 0.20) in a collection of ≥ 3 documents are
    flagged in-place — no new rows are created.

Re-runnable
-----------
Synthetic gap nodes are tagged with  metadata_json["is_synthetic_gap"] = True.
On each run, those rows are deleted and all is_gap flags are reset before the
three passes execute.

DB mapping
----------
  expected_topic  →  GapType.MISSING_LINK
  bridging        →  GapType.MISSING_LINK
  weak_coverage   →  GapType.UNDER_EXPLORED

Each gap stores actionable suggestions in  metadata_json["suggestions"].
"""
from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database_models import (
    Concept,
    Document,
    GapType,
    Relationship,
)
from app.services.llm_extractor import OllamaLLMService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GapDetectionResult:
    """Returned by GapDetector.detect_gaps()."""

    project_id: int
    expected_gaps_count: int
    bridging_gaps_count: int
    weak_coverage_count: int
    total_gaps: int
    suggestions: List[str]          # top-level actionable suggestions (≤ 20)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EXPECTED_TOPICS_PROMPT = """\
You are a research domain expert analyzing a collection of academic documents.

The following concepts have been identified in this collection:
{concept_list_with_descriptions}

These concepts span the domain of: {inferred_domain}

Based on your knowledge of this domain, what important related concepts are MISSING \
from this collection? These should be topics that a comprehensive understanding of \
this domain would require but are not covered.

For each missing concept provide:
1. concept_name: Clear name (2-5 words)
2. description: Why this concept is important for this domain (1-2 sentences)
3. related_to: Which existing concepts this would connect to (list of names)
4. importance: "critical" (fundamental gap), "important" (notable absence), \
or "nice_to_have" (would enhance completeness)

Return between 3 and {max_gaps} missing concepts. Do NOT suggest concepts that \
already appear in the list above.

Respond ONLY with valid JSON array — nothing else:
[{{"concept_name": "...", "description": "...", "related_to": ["..."], "importance": "critical"}}]\
"""

_EXPECTED_TOPICS_RETRY = """\
List {max_gaps} important concepts missing from this research collection.

Existing concepts: {concept_names_only}

Return ONLY a JSON array — nothing else:
[{{"concept_name": "Missing Topic", "description": "Why it matters in one sentence.", "related_to": ["Existing Concept"], "importance": "important"}}]\
"""

_BRIDGING_PROMPT = """\
Two related research clusters exist in a document collection but have no \
connecting research between them.

Cluster A — "{cluster_a_label}" — contains concepts like: {cluster_a_concepts}
Cluster B — "{cluster_b_label}" — contains concepts like: {cluster_b_concepts}

These clusters are semantically related (similarity: {similarity:.2f}) but no \
research in the collection bridges them.

What concept or research topic lies at the intersection of these two clusters? \
Suggest a single bridging concept that would connect them.

Respond ONLY in valid JSON — nothing else:
{{"concept_name": "...", "description": "How this bridges the two clusters in 1-2 sentences.", "importance": "critical or important or nice_to_have"}}\
"""

_BRIDGING_RETRY = """\
Suggest ONE bridging concept between:
  Cluster A: {cluster_a_label} (e.g. {cluster_a_concepts})
  Cluster B: {cluster_b_label} (e.g. {cluster_b_concepts})

Return ONLY JSON:
{{"concept_name": "Bridge Concept", "description": "Connects both clusters.", "importance": "important"}}\
"""


# ---------------------------------------------------------------------------
# GapDetector
# ---------------------------------------------------------------------------

class GapDetector:
    """
    Detects knowledge gaps in a project's concept graph.

    Call  ``await detector.detect_gaps(project_id, db)``  to run all three
    passes and persist the results.
    """

    # ---- tunables ----------------------------------------------------------
    WEAK_COVERAGE_THRESHOLD: float = 0.20   # coverage below this = weak
    WEAK_GENERALITY_MIN: float = 0.50       # generality above this = broad concept
    MIN_DOCS_FOR_WEAK_COVERAGE: int = 3     # skip pass 3 for tiny collections
    BRIDGE_SIM_THRESHOLD: float = 0.40      # min centroid cosine sim to flag a bridge gap
    MAX_EXPECTED_GAPS: int = 8              # max expected-topic gap concepts to create
    MAX_BRIDGING_PAIRS: int = 5             # max cluster pairs to generate bridge gaps for
    TOP_CONCEPTS_FOR_PROMPT: int = 25       # concept cap for the expected-topics prompt
    TOP_DOMAIN_CONCEPTS: int = 8            # highest-generality concepts used to infer domain
    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        self._llm = OllamaLLMService()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def detect_gaps(
        self,
        project_id: int,
        db: AsyncSession,
    ) -> GapDetectionResult:
        """
        Run all three gap detection passes and persist the results.

        Safe to call multiple times — previous synthetic gap nodes and gap
        flags are cleared before each run.
        """
        # Deferred import to avoid circular dependency (embedding ↔ llm_extractor)
        from app.services.embedding import OllamaEmbeddingService
        embedding_svc = OllamaEmbeddingService()

        # 0. Clear old results (re-runnable)
        logger.info("GapDetector: clearing previous gap data for project %d", project_id)
        await self._clear_old_gaps(project_id, db)

        # 1. Load all project data once
        logger.info("GapDetector: loading project data...")
        concepts = await self._load_concepts(project_id, db)
        relationships = await self._load_relationships(project_id, db)
        doc_count = await self._count_documents(project_id, db)

        if not concepts:
            logger.warning(
                "GapDetector: no concepts found for project %d — nothing to do",
                project_id,
            )
            return GapDetectionResult(project_id, 0, 0, 0, 0, [])

        logger.info(
            "GapDetector: %d concept(s), %d relationship(s), %d document(s)",
            len(concepts),
            len(relationships),
            doc_count,
        )

        all_suggestions: List[str] = []

        # ---- Pass 1: Expected Topic Gaps (LLM) ---------------------------
        logger.info("GapDetector: Pass 1 — expected topic gaps (LLM)...")
        expected_count, exp_sugg = await self._detect_expected_topics(
            concepts, project_id, db, embedding_svc
        )
        all_suggestions.extend(exp_sugg)
        logger.info(
            "GapDetector: Pass 1 done — created %d expected-topic gap node(s)",
            expected_count,
        )

        # ---- Pass 2: Bridging Gaps (graph topology + LLM) ---------------
        logger.info("GapDetector: Pass 2 — bridging gaps (graph + LLM)...")
        bridging_count, bridge_sugg = await self._detect_bridging_gaps(
            concepts, relationships, project_id, db
        )
        all_suggestions.extend(bridge_sugg)
        logger.info(
            "GapDetector: Pass 2 done — created %d bridging gap node(s)",
            bridging_count,
        )

        # ---- Pass 3: Weak Coverage (statistical, no LLM) ----------------
        logger.info("GapDetector: Pass 3 — weak coverage gaps (statistical)...")
        weak_count, weak_sugg = await self._detect_weak_coverage(
            concepts, doc_count, db
        )
        all_suggestions.extend(weak_sugg)
        logger.info(
            "GapDetector: Pass 3 done — flagged %d weak-coverage concept(s)",
            weak_count,
        )

        # ---- Persist everything ------------------------------------------
        await db.commit()

        total = expected_count + bridging_count + weak_count
        logger.info(
            "GapDetector: complete. expected=%d bridging=%d weak=%d total=%d",
            expected_count,
            bridging_count,
            weak_count,
            total,
        )

        return GapDetectionResult(
            project_id=project_id,
            expected_gaps_count=expected_count,
            bridging_gaps_count=bridging_count,
            weak_coverage_count=weak_count,
            total_gaps=total,
            suggestions=all_suggestions[:20],
        )

    # ------------------------------------------------------------------
    # Re-runnable: clear previous run
    # ------------------------------------------------------------------

    async def _clear_old_gaps(
        self,
        project_id: int,
        db: AsyncSession,
    ) -> None:
        """
        Delete synthetic gap Concept rows and reset is_gap flags on real
        concepts so the detector starts from a clean slate.

        Synthetic gap nodes carry  metadata_json["is_synthetic_gap"] = True.
        Real concepts that were flagged have their is_gap / gap_type / related
        metadata keys reset.
        """
        result = await db.execute(
            select(Concept).where(Concept.project_id == project_id)
        )
        all_concepts = result.scalars().all()

        synthetic_count = 0
        for c in all_concepts:
            meta = c.metadata_json or {}
            if meta.get("is_synthetic_gap"):
                await db.delete(c)
                synthetic_count += 1
            elif c.is_gap:
                # Reset gap flag on real concepts; preserve non-gap metadata
                c.is_gap = False
                c.gap_type = None
                cleaned = {
                    k: v
                    for k, v in meta.items()
                    if k not in {"gap_subtype", "suggestions", "related_to", "importance"}
                }
                c.metadata_json = cleaned or None

        if synthetic_count:
            logger.info(
                "GapDetector: deleted %d synthetic gap node(s)", synthetic_count
            )

        await db.flush()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def _load_concepts(
        self, project_id: int, db: AsyncSession
    ) -> List[Concept]:
        result = await db.execute(
            select(Concept).where(Concept.project_id == project_id)
        )
        return list(result.scalars().all())

    async def _load_relationships(
        self, project_id: int, db: AsyncSession
    ) -> List[Relationship]:
        # Join through Concept to scope relationships to this project
        result = await db.execute(
            select(Relationship)
            .join(Concept, Relationship.source_concept_id == Concept.id)
            .where(Concept.project_id == project_id)
        )
        return list(result.scalars().all())

    async def _count_documents(
        self, project_id: int, db: AsyncSession
    ) -> int:
        result = await db.execute(
            select(Document).where(Document.project_id == project_id)
        )
        return len(result.scalars().all())

    # ------------------------------------------------------------------
    # Pass 1: Expected Topic Gaps (LLM)
    # ------------------------------------------------------------------

    async def _detect_expected_topics(
        self,
        concepts: List[Concept],
        project_id: int,
        db: AsyncSession,
        embedding_svc: Any,
    ) -> Tuple[int, List[str]]:
        """
        Ask the LLM which domain topics are conspicuously absent from the
        collection.  Creates one synthetic Concept row per suggested gap.

        Returns (count_created, top_suggestions).
        """
        # Only consider real (non-synthetic) concepts
        real_concepts = [
            c for c in concepts
            if not (c.metadata_json or {}).get("is_synthetic_gap")
        ]
        if not real_concepts:
            return 0, []

        # Select the most representative concepts for the prompt
        # (highest coverage = most well-evidenced view of what IS in the collection)
        representative = sorted(
            real_concepts,
            key=lambda c: (c.coverage_score or 0.0),
            reverse=True,
        )[: self.TOP_CONCEPTS_FOR_PROMPT]

        # Format concept list
        concept_lines: List[str] = []
        for c in representative:
            desc = (c.description or "").strip()[:120]
            if desc:
                concept_lines.append(f"- {c.name}: {desc}")
            else:
                concept_lines.append(f"- {c.name}")
        concept_list_str = "\n".join(concept_lines)

        inferred_domain = self._infer_domain(real_concepts)
        concept_names_only = ", ".join(c.name for c in representative[:15])

        prompt = _EXPECTED_TOPICS_PROMPT.format(
            concept_list_with_descriptions=concept_list_str,
            inferred_domain=inferred_domain,
            max_gaps=self.MAX_EXPECTED_GAPS,
        )
        retry = _EXPECTED_TOPICS_RETRY.format(
            max_gaps=self.MAX_EXPECTED_GAPS,
            concept_names_only=concept_names_only,
        )

        logger.info(
            "GapDetector: calling LLM for expected topics (%d concepts in prompt)",
            len(representative),
        )

        success, raw = await self._llm._call_llm_json(prompt, retry_prompt=retry)
        if not success:
            logger.warning("GapDetector: LLM call for expected topics failed")
            return 0, []

        if not isinstance(raw, list):
            raw = [raw] if isinstance(raw, dict) else []

        # Guard against suggesting names that already exist
        existing_lower: Set[str] = {c.name.lower() for c in real_concepts}

        created = 0
        suggestions: List[str] = []

        for item in raw:
            if not isinstance(item, dict):
                continue

            name = str(item.get("concept_name", "")).strip()
            if not name or name.lower() in existing_lower:
                continue

            description = str(item.get("description", "")).strip()
            related_to: List[str] = [
                str(r).strip()
                for r in (item.get("related_to") or [])
                if r
            ][:5]
            importance = str(item.get("importance", "important")).lower()
            if importance not in {"critical", "important", "nice_to_have"}:
                importance = "important"

            # Embed the gap concept so it can be positioned on the map
            embed_text = f"{name} {description}".strip()
            embedding = await embedding_svc.embed_text(embed_text) if embed_text else None

            # Build actionable suggestions
            gap_suggestions = [
                f"Search for papers about: {name}",
                f"'{name}' is absent but relevant to the domain of "
                f"{inferred_domain.split(',')[0].strip()}",
            ]
            if related_to:
                gap_suggestions.append(
                    f"'{name}' would connect to existing concepts: "
                    f"{', '.join(related_to[:3])}"
                )

            gap_concept = Concept(
                project_id=project_id,
                name=name,
                description=description,
                generality_score=0.6,       # expected topics tend to be broad
                coverage_score=0.0,         # not covered at all
                consensus_score=None,
                embedding=embedding,
                is_gap=True,
                gap_type=GapType.MISSING_LINK,
                metadata_json={
                    "is_synthetic_gap": True,
                    "gap_subtype": "expected_topic",
                    "suggestions": gap_suggestions,
                    "related_to": related_to,
                    "importance": importance,
                },
            )
            db.add(gap_concept)
            existing_lower.add(name.lower())    # prevent within-run duplicates
            suggestions.append(f"Search for papers about: {name}")
            created += 1

            if created >= self.MAX_EXPECTED_GAPS:
                break

        await db.flush()
        return created, suggestions

    # ------------------------------------------------------------------
    # Pass 2: Bridging Gaps (graph topology + LLM)
    # ------------------------------------------------------------------

    async def _detect_bridging_gaps(
        self,
        concepts: List[Concept],
        relationships: List[Relationship],
        project_id: int,
        db: AsyncSession,
    ) -> Tuple[int, List[str]]:
        """
        Find clusters that are semantically related but have no inter-cluster
        Relationship rows.  For each such pair, the LLM suggests a bridging
        concept whose embedding is placed at the average of the two cluster
        centroids.

        Returns (count_created, top_suggestions).
        """
        # Build cluster membership index (real concepts with embeddings only)
        cluster_members: Dict[int, List[Concept]] = defaultdict(list)
        for c in concepts:
            if (
                c.cluster_label is not None
                and c.embedding is not None
                and not (c.metadata_json or {}).get("is_synthetic_gap")
            ):
                cluster_members[c.cluster_label].append(c)

        if len(cluster_members) < 2:
            logger.info(
                "GapDetector: fewer than 2 clusters — skipping bridging pass"
            )
            return 0, []

        # Index: concept_id → cluster_label
        concept_to_cluster: Dict[int, int] = {
            c.id: c.cluster_label
            for c in concepts
            if c.cluster_label is not None
        }

        # Build set of cluster pairs that already have inter-cluster relationships
        connected_pairs: Set[Tuple[int, int]] = set()
        for rel in relationships:
            src_cl = concept_to_cluster.get(rel.source_concept_id)
            tgt_cl = concept_to_cluster.get(rel.target_concept_id)
            if src_cl is not None and tgt_cl is not None and src_cl != tgt_cl:
                connected_pairs.add((min(src_cl, tgt_cl), max(src_cl, tgt_cl)))

        # Compute L2-normalized centroids and cluster head names
        cluster_labels = sorted(cluster_members.keys())
        centroids: Dict[int, np.ndarray] = {}
        cluster_head_names: Dict[int, str] = {}

        for lbl in cluster_labels:
            members = cluster_members[lbl]

            # Safely convert pgvector → numpy rows
            emb_rows: List[List[float]] = []
            for c in members:
                try:
                    emb_rows.append(list(c.embedding))
                except (TypeError, ValueError):
                    continue

            if not emb_rows:
                continue

            emb_matrix = np.array(emb_rows, dtype=np.float32)
            centroid = emb_matrix.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-9:
                centroid /= norm
            centroids[lbl] = centroid

            # Cluster representative = highest-generality member
            head = max(members, key=lambda c: c.generality_score or 0.0)
            cluster_head_names[lbl] = head.name

        # Identify unconnected pairs with high centroid similarity
        bridge_candidates: List[Tuple[int, int, float]] = []
        for i, lbl_a in enumerate(cluster_labels):
            if lbl_a not in centroids:
                continue
            for lbl_b in cluster_labels[i + 1 :]:
                if lbl_b not in centroids:
                    continue
                pair = (min(lbl_a, lbl_b), max(lbl_a, lbl_b))
                if pair in connected_pairs:
                    continue
                sim = float(np.dot(centroids[lbl_a], centroids[lbl_b]))
                if sim >= self.BRIDGE_SIM_THRESHOLD:
                    bridge_candidates.append((lbl_a, lbl_b, sim))

        # Keep highest-similarity pairs first; cap to MAX_BRIDGING_PAIRS
        bridge_candidates.sort(key=lambda x: x[2], reverse=True)
        bridge_candidates = bridge_candidates[: self.MAX_BRIDGING_PAIRS]

        if not bridge_candidates:
            logger.info(
                "GapDetector: no unconnected cluster pairs above "
                "similarity threshold %.2f",
                self.BRIDGE_SIM_THRESHOLD,
            )
            return 0, []

        logger.info(
            "GapDetector: %d bridge gap candidate(s) found", len(bridge_candidates)
        )

        created = 0
        suggestions: List[str] = []

        for lbl_a, lbl_b, sim in bridge_candidates:
            head_a = cluster_head_names.get(lbl_a, f"Cluster {lbl_a}")
            head_b = cluster_head_names.get(lbl_b, f"Cluster {lbl_b}")

            # Up to 4 representative concept names per cluster for the prompt
            sample_a = ", ".join(c.name for c in cluster_members[lbl_a][:4])
            sample_b = ", ".join(c.name for c in cluster_members[lbl_b][:4])

            prompt = _BRIDGING_PROMPT.format(
                cluster_a_label=head_a,
                cluster_a_concepts=sample_a,
                cluster_b_label=head_b,
                cluster_b_concepts=sample_b,
                similarity=sim,
            )
            retry = _BRIDGING_RETRY.format(
                cluster_a_label=head_a,
                cluster_a_concepts=sample_a,
                cluster_b_label=head_b,
                cluster_b_concepts=sample_b,
            )

            success, parsed = await self._llm._call_llm_json(
                prompt, retry_prompt=retry
            )
            if not success or not isinstance(parsed, dict):
                logger.warning(
                    "GapDetector: LLM bridging call failed for clusters "
                    "%d (%s) ↔ %d (%s)",
                    lbl_a,
                    head_a,
                    lbl_b,
                    head_b,
                )
                continue

            name = str(parsed.get("concept_name", "")).strip()
            description = str(parsed.get("description", "")).strip()
            importance = str(parsed.get("importance", "important")).lower()
            if not name:
                continue
            if importance not in {"critical", "important", "nice_to_have"}:
                importance = "important"

            # Bridge embedding = average of the two cluster centroids (L2-normalized)
            # This positions the gap concept between the clusters on the semantic map
            bridge_vec = (centroids[lbl_a] + centroids[lbl_b]) / 2.0
            bridge_norm = float(np.linalg.norm(bridge_vec))
            if bridge_norm > 1e-9:
                bridge_vec /= bridge_norm
            bridge_embedding = bridge_vec.tolist()

            gap_suggestions = [
                f"This gap connects '{head_a}' and '{head_b}'",
                f"Search for papers about: {name}",
                f"'{name}' may bridge the '{head_a}' and '{head_b}' research clusters",
            ]

            gap_concept = Concept(
                project_id=project_id,
                name=name,
                description=description,
                generality_score=0.5,
                coverage_score=0.0,
                consensus_score=None,
                embedding=bridge_embedding,
                is_gap=True,
                gap_type=GapType.MISSING_LINK,
                metadata_json={
                    "is_synthetic_gap": True,
                    "gap_subtype": "bridging",
                    "suggestions": gap_suggestions,
                    "related_to": [head_a, head_b],
                    "importance": importance,
                    "bridge_clusters": [lbl_a, lbl_b],
                    "cluster_similarity": round(sim, 4),
                },
            )
            db.add(gap_concept)
            suggestions.append(f"This gap connects '{head_a}' and '{head_b}'")
            created += 1

        await db.flush()
        return created, suggestions

    # ------------------------------------------------------------------
    # Pass 3: Weak Coverage Gaps (statistical)
    # ------------------------------------------------------------------

    async def _detect_weak_coverage(
        self,
        concepts: List[Concept],
        doc_count: int,
        db: AsyncSession,
    ) -> Tuple[int, List[str]]:
        """
        Flag existing broad concepts that are barely covered by the collection.

        Criteria:
          - generality_score > WEAK_GENERALITY_MIN  (it is a broad/important topic)
          - coverage_score   < WEAK_COVERAGE_THRESHOLD  (few documents mention it)
          - doc_count        >= MIN_DOCS_FOR_WEAK_COVERAGE  (collection is large enough)

        Does NOT create new rows — marks the existing Concept in place.

        Returns (count_flagged, top_suggestions).
        """
        if doc_count < self.MIN_DOCS_FOR_WEAK_COVERAGE:
            logger.info(
                "GapDetector: skipping weak-coverage pass "
                "(collection has %d doc(s); need ≥ %d)",
                doc_count,
                self.MIN_DOCS_FOR_WEAK_COVERAGE,
            )
            return 0, []

        flagged = 0
        suggestions: List[str] = []

        for concept in concepts:
            # Never flag synthetic gap nodes from this run
            if (concept.metadata_json or {}).get("is_synthetic_gap"):
                continue

            coverage = concept.coverage_score or 0.0
            generality = concept.generality_score or 0.0

            if (
                coverage < self.WEAK_COVERAGE_THRESHOLD
                and generality > self.WEAK_GENERALITY_MIN
            ):
                concept.is_gap = True
                concept.gap_type = GapType.UNDER_EXPLORED

                # Estimate rough document count from coverage_score
                approx_doc_count = max(1, round(coverage * doc_count))
                gap_suggestions = [
                    f"Search for papers about: {concept.name}",
                    f"Only ~{approx_doc_count} of {doc_count} document(s) cover "
                    f"'{concept.name}' — consider adding more sources",
                ]

                # Merge gap metadata into existing metadata_json
                existing_meta = dict(concept.metadata_json or {})
                existing_meta.update(
                    {
                        "gap_subtype": "weak_coverage",
                        "suggestions": gap_suggestions,
                        "related_to": [],
                        "importance": "important",
                    }
                )
                concept.metadata_json = existing_meta

                # One representative suggestion per concept for the summary
                suggestions.append(gap_suggestions[1])
                flagged += 1

        return flagged, suggestions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_domain(self, concepts: List[Concept]) -> str:
        """
        Return a short, human-readable domain description by joining the names
        of the most general (highest generality_score) concepts.

        Example output: "machine learning, neural networks, computer vision"
        """
        real = [
            c for c in concepts
            if not (c.metadata_json or {}).get("is_synthetic_gap")
        ]
        top = sorted(
            real,
            key=lambda c: c.generality_score or 0.0,
            reverse=True,
        )[: self.TOP_DOMAIN_CONCEPTS]
        return ", ".join(c.name for c in top) if top else "research"


# ---------------------------------------------------------------------------
# Backward-compatible stubs
# (kept so any lingering code that imports the old API doesn't crash)
# ---------------------------------------------------------------------------

__all__ = ["GapDetector", "GapDetectionResult"]
