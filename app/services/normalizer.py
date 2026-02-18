"""
Concept normalisation and deduplication service.

Pipeline (normalize_project)
-----------------------------
1. Load all concepts for the project from the database.
2. Group by exact normalised name  → guaranteed duplicates.
3. Build one representative embedding per name-group (average).
4. Run AgglomerativeClustering (cosine distance, average linkage) on the
   representatives to find semantically similar groups.
5. For each cluster:
   - Select the canonical name (most frequent → shortest → alphabetical).
   - Merge descriptions.
   - Average all embeddings and re-normalise to unit length.
   - Compute coverage_score = distinct docs with a claim / total docs.
   - Persist: update the canonical Concept row, remap all FK references
     (claims, relationships, parent_concept_id) from merged IDs to the
     canonical ID, then delete the merged rows.
6. Update coverage scores for singleton concepts.
7. Return NormalizationResult.

Backward-compatible API
------------------------
The original normalize_concepts / find_canonical_concept /
merge_concept_hierarchies methods are kept unchanged so that the existing
concepts.py background-task code continues to work without modification.
"""
from __future__ import annotations

import dataclasses
import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database_models import Claim, Concept, Document, Relationship
from app.utils.helpers import clean_concept_name, cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NormalizationResult:
    """Summary of what normalize_project did."""

    project_id: int
    total_concepts_before: int
    canonical_concepts_after: int
    merged_count: int
    # One entry per merged group (singleton groups are excluded)
    groups_merged: List[Dict[str, Any]]
    # variant_name -> canonical_name for every alias
    alias_map: Dict[str, str]


# ---------------------------------------------------------------------------
# Pure vector helpers (module-level so they can be reused)
# ---------------------------------------------------------------------------

def _to_float_list(value: Any) -> List[float]:
    """Safely convert a pgvector / numpy / list value to List[float]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [float(x) for x in value]
    try:
        return [float(x) for x in value]
    except Exception:
        return []


def _normalize_vec(vector: List[float]) -> List[float]:
    """Return a unit-length copy of *vector*."""
    mag = math.sqrt(sum(x * x for x in vector))
    if mag < 1e-10:
        return vector
    return [x / mag for x in vector]


def _average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """Element-wise mean of equal-length float lists, then L2-normalised."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    avg = [
        sum(e[i] for e in embeddings) / len(embeddings)
        for i in range(dim)
    ]
    return _normalize_vec(avg)


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class ConceptNormalizer:
    """
    Concept normalisation combining exact-name matching and embedding-based
    clustering (AgglomerativeClustering from scikit-learn).

    Parameters
    ----------
    similarity_threshold:
        Cosine-similarity floor for treating two concepts as equivalent.
        Defaults to ``settings.CONCEPT_SIMILARITY_THRESHOLD`` (0.85).
        Override per-instance for testing.
    """

    def __init__(self, similarity_threshold: Optional[float] = None) -> None:
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.CONCEPT_SIMILARITY_THRESHOLD
        )

    # ------------------------------------------------------------------
    # Public: DB-backed full pipeline
    # ------------------------------------------------------------------

    async def normalize_project(
        self,
        project_id: int,
        db: AsyncSession,
    ) -> NormalizationResult:
        """
        Run the full normalisation pipeline for *project_id*.

        All DB mutations are committed before returning.
        Safe to call multiple times (idempotent at the concept level).
        """
        # 1. Count documents (needed for coverage scores)
        doc_count_res = await db.execute(
            select(func.count(Document.id)).where(
                Document.project_id == project_id
            )
        )
        total_docs = int(doc_count_res.scalar_one() or 0)

        # 2. Load all concepts
        concept_res = await db.execute(
            select(Concept).where(Concept.project_id == project_id)
        )
        all_concepts: List[Concept] = list(concept_res.scalars().all())
        total_before = len(all_concepts)

        if total_before == 0:
            logger.info("normalize_project(%d): no concepts to normalise", project_id)
            return NormalizationResult(
                project_id=project_id,
                total_concepts_before=0,
                canonical_concepts_after=0,
                merged_count=0,
                groups_merged=[],
                alias_map={},
            )

        # 3. Effective similarity threshold (relax for small projects)
        if total_before < settings.CONCEPT_SMALL_PROJECT_THRESHOLD:
            threshold = settings.CONCEPT_SIMILARITY_THRESHOLD_SMALL
            logger.info(
                "normalize_project(%d): small project (%d concepts) — "
                "using relaxed threshold %.2f",
                project_id,
                total_before,
                threshold,
            )
        else:
            threshold = self.similarity_threshold

        # 4. First pass — group by exact normalised name
        name_groups: Dict[str, List[Concept]] = defaultdict(list)
        nameless: List[Concept] = []
        for c in all_concepts:
            norm = clean_concept_name(c.name or "")
            if norm:
                name_groups[norm].append(c)
            else:
                nameless.append(c)

        logger.info(
            "normalize_project(%d): %d concepts → %d unique normalised names",
            project_id,
            total_before,
            len(name_groups),
        )

        # 5. Build one representative per name-group
        #    (norm_name, avg_embedding | None, [Concept ...])
        NameGroup = Tuple[str, Optional[List[float]], List[Concept]]
        groups: List[NameGroup] = []
        for norm_name, concepts in name_groups.items():
            valid_embs = [
                _to_float_list(c.embedding)
                for c in concepts
                if c.embedding is not None
            ]
            avg_emb = _average_embeddings(valid_embs) if valid_embs else None
            groups.append((norm_name, avg_emb, concepts))

        # 6. Split by embedding availability
        with_emb: List[NameGroup] = [(n, e, cs) for n, e, cs in groups if e]
        without_emb: List[NameGroup] = [(n, e, cs) for n, e, cs in groups if not e]

        # 7. Cluster the groups that have embeddings
        #    Each element in with_emb becomes one "super-node"; the cluster
        #    label tells us which super-nodes to merge together.
        final_groups: List[List[Concept]] = []

        if len(with_emb) >= 2:
            emb_matrix = [e for _, e, _ in with_emb]
            labels = self._agglomerative_cluster(emb_matrix, threshold)

            # Collect super-nodes sharing the same label
            cluster_map: Dict[int, List[Concept]] = defaultdict(list)
            for label, (norm_name, _emb, concepts) in zip(labels, with_emb):
                cluster_map[label].extend(concepts)

            # Log merges at cluster level
            for label, concept_list in cluster_map.items():
                names_in_cluster = sorted({c.name for c in concept_list})
                if len(names_in_cluster) > 1:
                    logger.info(
                        "Cluster %d will merge: %s",
                        label,
                        names_in_cluster,
                    )
                final_groups.append(concept_list)

        elif len(with_emb) == 1:
            _n, _e, concepts = with_emb[0]
            final_groups.append(concepts)

        # Concepts without embeddings are kept as individual groups
        for _n, _e, concepts in without_emb:
            final_groups.append(concepts)

        # Nameless concepts are each their own singleton
        for c in nameless:
            final_groups.append([c])

        # 8. Process each group — merge + persist
        groups_merged_info: List[Dict[str, Any]] = []
        alias_map: Dict[str, str] = {}
        canonical_ids: List[int] = []

        for group in final_groups:
            if len(group) == 1:
                # Singleton — update coverage score but don't merge
                c = group[0]
                coverage = await self._coverage_score([c.id], total_docs, db)
                c.coverage_score = coverage
                canonical_ids.append(c.id)
                continue

            # --- Multiple concepts in this group: merge them ---
            canonical_name = self._select_canonical_name(group)
            canonical_record = self._pick_record_to_keep(group)
            merged_records = [c for c in group if c.id != canonical_record.id]

            all_names = [c.name for c in group if c.name]
            all_descs = [c.description for c in group if c.description]
            valid_embs = [
                _to_float_list(c.embedding)
                for c in group
                if c.embedding is not None
            ]

            merged_desc = self._merge_descriptions(all_descs)
            merged_emb = _average_embeddings(valid_embs) if valid_embs else None
            aliases = sorted({n for n in all_names if n != canonical_name})
            all_ids = [c.id for c in group]
            coverage = await self._coverage_score(all_ids, total_docs, db)

            # Remap all FK references before deletion
            for merged_c in merged_records:
                await self._remap_concept_references(
                    merged_c.id, canonical_record.id, db
                )
            await db.flush()

            # Update canonical record
            old_meta = dict(canonical_record.metadata_json or {})
            old_meta["aliases"] = aliases
            old_meta["merged_from_ids"] = [c.id for c in merged_records]
            old_meta["is_canonical"] = True

            canonical_record.name = canonical_name
            canonical_record.description = merged_desc
            canonical_record.coverage_score = coverage
            if merged_emb:
                canonical_record.embedding = merged_emb
            canonical_record.metadata_json = old_meta

            # Delete merged rows via raw SQL to avoid ORM cascade surprises
            for merged_c in merged_records:
                await db.execute(
                    sql_delete(Concept).where(Concept.id == merged_c.id)
                )
            await db.flush()

            canonical_ids.append(canonical_record.id)

            # Track merge info
            for name in all_names:
                alias_map[name] = canonical_name

            merge_reason = (
                "exact name match"
                if len({clean_concept_name(c.name or "") for c in group}) == 1
                else f"embedding cosine similarity ≥ {threshold:.2f}"
            )

            logger.info(
                "Merged %d concepts → '%s' (aliases: %s, reason: %s)",
                len(group),
                canonical_name,
                aliases,
                merge_reason,
            )

            groups_merged_info.append({
                "canonical_id": canonical_record.id,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "merged_count": len(merged_records),
                "coverage_score": round(coverage, 4),
                "reason": merge_reason,
            })

        await db.commit()

        canonical_after = len(canonical_ids)
        merged_total = total_before - canonical_after

        logger.info(
            "normalize_project(%d): %d → %d concepts (%d merged)",
            project_id,
            total_before,
            canonical_after,
            merged_total,
        )

        return NormalizationResult(
            project_id=project_id,
            total_concepts_before=total_before,
            canonical_concepts_after=canonical_after,
            merged_count=merged_total,
            groups_merged=groups_merged_info,
            alias_map=alias_map,
        )

    # ------------------------------------------------------------------
    # Private: clustering
    # ------------------------------------------------------------------

    def _agglomerative_cluster(
        self,
        embeddings: List[List[float]],
        threshold: float,
    ) -> List[int]:
        """
        Cluster *embeddings* using AgglomerativeClustering with cosine
        distance and average linkage.

        distance_threshold = 1 - similarity_threshold, so concepts within
        that cosine distance are merged into the same cluster.

        Falls back to one-cluster-per-item if sklearn is unavailable or
        clustering fails.
        """
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            logger.error(
                "scikit-learn is not installed — falling back to no embedding clustering. "
                "Run: pip install scikit-learn"
            )
            return list(range(len(embeddings)))

        n = len(embeddings)
        if n <= 1:
            return [0] * n

        try:
            X = np.array(embeddings, dtype=np.float64)
            # Ensure unit-length rows (cosine distance is defined for unit vectors)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            X = X / norms

            distance_threshold = 1.0 - threshold
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=distance_threshold,
            )
            labels: List[int] = clustering.fit_predict(X).tolist()
            n_clusters = len(set(labels))
            logger.info(
                "_agglomerative_cluster: %d groups → %d clusters "
                "(threshold=%.2f, distance=%.2f)",
                n,
                n_clusters,
                threshold,
                distance_threshold,
            )
            return labels

        except Exception as exc:
            logger.error(
                "_agglomerative_cluster: clustering failed (%s) — "
                "treating every group as its own cluster",
                exc,
            )
            return list(range(n))

    # ------------------------------------------------------------------
    # Private: canonical selection
    # ------------------------------------------------------------------

    def _select_canonical_name(self, group: List[Concept]) -> str:
        """
        Choose the canonical name for a group of merged concepts.

        Priority
        --------
        1. Name that appears most frequently across the group.
        2. If tied: shorter name (less jargon/abbreviation risk).
        3. If still tied: alphabetically first.
        """
        name_counts: Dict[str, int] = defaultdict(int)
        for c in group:
            if c.name:
                name_counts[c.name] += 1

        if not name_counts:
            return "Unknown Concept"

        max_count = max(name_counts.values())
        candidates = [n for n, cnt in name_counts.items() if cnt == max_count]
        candidates.sort(key=lambda n: (len(n), n))
        canonical = candidates[0]

        logger.debug(
            "_select_canonical_name: chose '%s' from %s",
            canonical,
            sorted(name_counts.keys()),
        )
        return canonical

    def _pick_record_to_keep(self, group: List[Concept]) -> Concept:
        """
        Choose which DB row to keep as the canonical record.

        Prefer the record that already has an embedding; break ties by
        lowest id (oldest, hence most stable).
        """
        with_emb = [c for c in group if c.embedding is not None]
        pool = with_emb if with_emb else group
        return min(pool, key=lambda c: c.id)

    def _merge_descriptions(self, descriptions: List[str]) -> str:
        """
        Combine multiple concept descriptions into one.

        Strategy: deduplicate, sort by length descending, join the two
        most informative ones if the secondary adds meaningful content.
        """
        unique: List[str] = []
        seen: Set[str] = set()
        for desc in descriptions:
            stripped = (desc or "").strip()
            if stripped and stripped not in seen:
                unique.append(stripped)
                seen.add(stripped)

        if not unique:
            return ""
        if len(unique) == 1:
            return unique[0]

        # Sort longest first
        unique.sort(key=len, reverse=True)
        primary = unique[0]

        # Append shorter unique descriptions if they add substance
        extras = [d for d in unique[1:] if len(d) > 40]
        if not extras:
            return primary

        combined = primary + "; " + "; ".join(extras)
        # Hard cap to keep descriptions readable
        return combined[:600] if len(combined) > 600 else combined

    # ------------------------------------------------------------------
    # Private: coverage
    # ------------------------------------------------------------------

    async def _coverage_score(
        self,
        concept_ids: List[int],
        total_docs: int,
        db: AsyncSession,
    ) -> float:
        """
        coverage_score = distinct documents that have at least one claim
                         linking to any concept in *concept_ids*
                         ÷ total documents in the project.
        """
        if not concept_ids or total_docs == 0:
            return 0.0

        res = await db.execute(
            select(func.count(func.distinct(Claim.document_id))).where(
                Claim.concept_id.in_(concept_ids)
            )
        )
        doc_count = int(res.scalar_one() or 0)
        return min(1.0, doc_count / total_docs)

    # ------------------------------------------------------------------
    # Private: DB remapping
    # ------------------------------------------------------------------

    async def _remap_concept_references(
        self,
        old_id: int,
        new_id: int,
        db: AsyncSession,
    ) -> None:
        """
        Update every FK that points at *old_id* to point at *new_id*.

        Tables touched: claims, relationships (source + target),
        concepts (parent_concept_id).

        Must be called — and flushed — BEFORE deleting the old Concept row
        so that DB-level ON DELETE CASCADE cannot remove remapped rows.
        """
        # Claims
        await db.execute(
            update(Claim)
            .where(Claim.concept_id == old_id)
            .values(concept_id=new_id)
        )

        # Relationships — source side
        await db.execute(
            update(Relationship)
            .where(Relationship.source_concept_id == old_id)
            .values(source_concept_id=new_id)
        )

        # Relationships — target side
        await db.execute(
            update(Relationship)
            .where(Relationship.target_concept_id == old_id)
            .values(target_concept_id=new_id)
        )

        # Parent-child hierarchy
        await db.execute(
            update(Concept)
            .where(Concept.parent_concept_id == old_id)
            .values(parent_concept_id=new_id)
        )

        logger.debug(
            "_remap_concept_references: %d → %d (claims, relationships, "
            "parent refs updated)",
            old_id,
            new_id,
        )

    # ------------------------------------------------------------------
    # Backward-compatible in-memory API
    # (used by concepts.py process_concept_extraction background task)
    # ------------------------------------------------------------------

    async def normalize_concepts(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalise and deduplicate an in-memory list of concept dicts.

        Kept for backward compatibility with the legacy extraction background
        task in concepts.py.  Does NOT touch the database.
        """
        if not concepts:
            return []

        # Pass 1: exact name matching
        name_to_group: Dict[str, List[Dict]] = defaultdict(list)
        for c in concepts:
            norm = clean_concept_name(c.get("name", ""))
            if norm:
                name_to_group[norm].append(c)

        merged: List[Dict[str, Any]] = []
        for _norm, grp in name_to_group.items():
            if len(grp) == 1:
                merged.append(grp[0])
            else:
                merged.append(await self._merge_dicts(grp))

        logger.info(
            "normalize_concepts (in-memory): %d → %d after exact matching",
            len(concepts),
            len(merged),
        )

        # Pass 2: embedding similarity
        final = await self._dedup_by_embedding(merged)
        logger.info(
            "normalize_concepts (in-memory): %d → %d after embedding dedup",
            len(merged),
            len(final),
        )
        return final

    async def _merge_dicts(
        self,
        group: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge a list of concept dicts into one (in-memory, no DB)."""
        base = group[0].copy()

        # Longest description wins
        descs = [c.get("description", "") for c in group if c.get("description")]
        if descs:
            base["description"] = max(descs, key=len)

        # Average numeric score fields
        for field in ("generality_score", "coverage_score", "consensus_score"):
            scores = [c[field] for c in group if c.get(field) is not None]
            if scores:
                base[field] = sum(scores) / len(scores)

        # Merge metadata
        meta: Dict[str, Any] = {}
        for c in group:
            if c.get("metadata_json"):
                meta.update(c["metadata_json"])
        if meta:
            base["metadata_json"] = meta

        return base

    async def _dedup_by_embedding(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deduplicate an in-memory concept list by embedding cosine similarity."""
        with_emb = [c for c in concepts if c.get("embedding") is not None]
        without_emb = [c for c in concepts if c.get("embedding") is None]

        if not with_emb:
            return concepts

        visited: Set[int] = set()
        result: List[Dict[str, Any]] = []

        for i, c1 in enumerate(with_emb):
            if i in visited:
                continue
            group = [c1]
            visited.add(i)

            for j, c2 in enumerate(with_emb[i + 1 :], start=i + 1):
                if j in visited:
                    continue
                try:
                    sim = cosine_similarity(
                        c1.get("embedding", []),
                        c2.get("embedding", []),
                    )
                except ValueError:
                    sim = 0.0
                if sim >= self.similarity_threshold:
                    group.append(c2)
                    visited.add(j)

            if len(group) > 1:
                merged = await self._merge_dicts(group)
                logger.debug(
                    "_dedup_by_embedding: merged %d concepts around '%s'",
                    len(group),
                    c1.get("name"),
                )
                result.append(merged)
            else:
                result.append(c1)

        result.extend(without_emb)
        return result

    async def find_canonical_concept(
        self,
        concept_name: str,
        existing_concepts: List[Dict[str, Any]],
        embedding: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the canonical entry for *concept_name* from *existing_concepts*,
        or ``None`` if not found.  Kept for backward compatibility.
        """
        norm = clean_concept_name(concept_name)

        for c in existing_concepts:
            if clean_concept_name(c.get("name", "")) == norm:
                return c

        if embedding:
            for c in existing_concepts:
                if c.get("embedding"):
                    try:
                        sim = cosine_similarity(embedding, c["embedding"])
                    except ValueError:
                        sim = 0.0
                    if sim >= self.similarity_threshold:
                        logger.debug(
                            "find_canonical_concept: '%s' → '%s' (sim=%.2f)",
                            concept_name,
                            c.get("name"),
                            sim,
                        )
                        return c

        return None

    async def merge_concept_hierarchies(
        self,
        concepts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Assign parent_concept_id based on generality_score and embedding
        similarity.  Kept for backward compatibility.
        """
        sorted_cs = sorted(
            concepts,
            key=lambda x: x.get("generality_score", 0.5),
            reverse=True,
        )

        for i, specific in enumerate(sorted_cs):
            if specific.get("parent_concept_id"):
                continue

            for general in sorted_cs[:i]:
                if (
                    general.get("generality_score", 0)
                    > specific.get("generality_score", 0) + 0.2
                ):
                    if specific.get("embedding") and general.get("embedding"):
                        try:
                            sim = cosine_similarity(
                                specific["embedding"], general["embedding"]
                            )
                        except ValueError:
                            sim = 0.0
                        if sim >= 0.6:
                            specific["parent_concept_id"] = general.get("id")
                            logger.debug(
                                "merge_concept_hierarchies: '%s' → '%s'",
                                specific.get("name"),
                                general.get("name"),
                            )
                            break

        return sorted_cs
