"""
Clustering service using HDBSCAN for concept organisation.

Public API
----------
ConceptClusterer.cluster_project(project_id, db)  → ClusteringResult
ClusteringService                                  → alias (backward compat)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database_models import Concept, Claim, Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClusterMember:
    concept_id: int
    name: str
    generality_score: float
    coverage_score: float
    is_gap: bool
    gap_type: Optional[str]
    is_cluster_head: bool


@dataclass
class ClusterInfo:
    cluster_id: int
    label: str                      # name of the cluster-head concept
    parent_concept_id: Optional[int]
    parent_concept_name: Optional[str]
    members: List[ClusterMember]
    size: int
    avg_generality: float
    cohesion: float                 # avg cosine similarity of members to centroid


@dataclass
class ClusteringResult:
    project_id: int
    total_concepts: int
    clustered_concepts: int
    noise_reassigned: int
    num_clusters: int
    clusters: List[ClusterInfo]
    bridge_relationships: List[Dict[str, Any]]
    algorithm_used: str


# ---------------------------------------------------------------------------
# ConceptClusterer
# ---------------------------------------------------------------------------

class ConceptClusterer:
    """
    HDBSCAN-based concept clusterer with KMeans fallback.

    Main entry point: ``cluster_project(project_id, db)``

    Backward-compat shim methods ``cluster_concepts()`` and
    ``detect_hierarchy()`` are provided for callers that pass raw embeddings.
    """

    def __init__(
        self,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
    ) -> None:
        self.min_cluster_size = min_cluster_size or settings.HDBSCAN_MIN_CLUSTER_SIZE
        self.min_samples = min_samples or settings.HDBSCAN_MIN_SAMPLES

    # ------------------------------------------------------------------
    # Public DB-backed pipeline
    # ------------------------------------------------------------------

    async def cluster_project(
        self, project_id: int, db: AsyncSession
    ) -> ClusteringResult:
        """
        Full clustering pipeline for a project.

        Steps
        -----
        1.  Load concepts that have embeddings.
        2.  Compute per-concept doc/claim frequencies (for generality scoring).
        3.  L2-normalise embeddings.
        4.  Run HDBSCAN with adaptive min_cluster_size; fall back to KMeans.
        5.  Reassign noise points (-1) to nearest cluster centroid.
        6.  Recompute generality scores (doc_freq, claim_freq, specificity,
            cluster centrality).
        7.  Build 2-level hierarchy: highest-generality concept → parent,
            rest → direct children.
        8.  Detect cross-cluster bridge relationships (cosine > threshold).
        9.  Persist cluster_label, parent_concept_id, generality_score to DB
            and commit.
        10. Build and return ClusteringResult.
        """
        # --- 1. Load concepts -----------------------------------------------
        concept_res = await db.execute(
            select(Concept)
            .where(Concept.project_id == project_id)
            .where(Concept.embedding.isnot(None))
        )
        concepts: List[Concept] = list(concept_res.scalars().all())
        n = len(concepts)

        if n < 2:
            logger.warning(
                "cluster_project: only %d concept(s) with embeddings — skipping", n
            )
            return ClusteringResult(
                project_id=project_id,
                total_concepts=n,
                clustered_concepts=0,
                noise_reassigned=0,
                num_clusters=0,
                clusters=[],
                bridge_relationships=[],
                algorithm_used="none",
            )

        concept_ids = [c.id for c in concepts]

        # --- 2. Claim / document frequencies --------------------------------
        claim_count_res = await db.execute(
            select(Claim.concept_id, func.count(Claim.id).label("cnt"))
            .where(Claim.concept_id.in_(concept_ids))
            .group_by(Claim.concept_id)
        )
        claim_counts: Dict[int, int] = {r.concept_id: r.cnt for r in claim_count_res}

        doc_count_res = await db.execute(
            select(
                Claim.concept_id,
                func.count(Claim.document_id.distinct()).label("cnt"),
            )
            .where(Claim.concept_id.in_(concept_ids))
            .group_by(Claim.concept_id)
        )
        doc_counts: Dict[int, int] = {r.concept_id: r.cnt for r in doc_count_res}

        total_docs_res = await db.execute(
            select(func.count(Document.id)).where(Document.project_id == project_id)
        )
        total_docs: int = int(total_docs_res.scalar() or 1)
        max_claims: int = max(claim_counts.values(), default=1)

        # --- 3. L2-normalise embeddings -------------------------------------
        raw = np.array([list(c.embedding) for c in concepts], dtype=float)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb: np.ndarray = raw / norms  # (n, dim) — unit vectors

        # --- 4. Clustering --------------------------------------------------
        # Adaptive min_cluster_size: grow with dataset but never < configured floor
        adaptive_mcs = max(self.min_cluster_size, n // 10) if n >= 10 else self.min_cluster_size
        adaptive_mcs = max(2, adaptive_mcs)
        labels, algorithm_used = self._run_clustering(emb, adaptive_mcs)

        # --- 5. Reassign noise ----------------------------------------------
        noise_before = int(sum(1 for l in labels if l == -1))
        labels = self._reassign_noise(labels, emb)

        # --- 6. Cluster centroids -------------------------------------------
        label_arr = np.array(labels, dtype=int)
        unique_labels = sorted(set(labels))
        centroids: Dict[int, np.ndarray] = {}
        for lbl in unique_labels:
            mask = label_arr == lbl
            centroids[lbl] = emb[mask].mean(axis=0)

        # --- 7. Recompute generality scores ---------------------------------
        generality: Dict[int, float] = {}
        for i, concept in enumerate(concepts):
            lbl = labels[i]
            doc_freq = doc_counts.get(concept.id, 0) / total_docs
            claim_freq = claim_counts.get(concept.id, 0) / max_claims
            # LLM-provided generality (0 = highly specific, 1 = highly general)
            specificity = float(concept.generality_score or 0.5)
            # Centrality: cosine sim to centroid (unit vecs → dot product)
            centrality = max(0.0, float(np.dot(emb[i], centroids[lbl])))
            g = (
                0.4 * doc_freq
                + 0.3 * claim_freq
                + 0.2 * specificity
                + 0.1 * centrality
            )
            generality[concept.id] = round(g, 4)

        # --- 8. Build 2-level hierarchy -------------------------------------
        cluster_groups: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            cluster_groups.setdefault(lbl, []).append(i)

        # parent_map: concept_id → parent concept_id (None = cluster head)
        parent_map: Dict[int, Optional[int]] = {}
        for lbl, indices in cluster_groups.items():
            sorted_i = sorted(
                indices, key=lambda i: generality[concepts[i].id], reverse=True
            )
            head_concept = concepts[sorted_i[0]]
            parent_map[head_concept.id] = None  # cluster head has no parent
            for child_i in sorted_i[1:]:
                parent_map[concepts[child_i].id] = head_concept.id

        # --- 9. Bridge relationships ----------------------------------------
        bridge_threshold: float = getattr(
            settings, "CLUSTER_BRIDGE_THRESHOLD", 0.60
        )
        bridges = self._detect_bridges(concepts, emb, labels, bridge_threshold)

        # --- 10. Persist to DB ----------------------------------------------
        for i, concept in enumerate(concepts):
            concept.cluster_label = int(labels[i])
            concept.generality_score = generality[concept.id]
            concept.parent_concept_id = parent_map.get(concept.id)

        await db.commit()

        # --- 11. Build ClusteringResult -------------------------------------
        cluster_infos: List[ClusterInfo] = []
        for lbl in unique_labels:
            indices = cluster_groups[lbl]
            head_id: Optional[int] = None
            head_name: Optional[str] = None
            members: List[ClusterMember] = []

            for i in indices:
                c = concepts[i]
                is_head = parent_map.get(c.id) is None
                if is_head:
                    head_id = c.id
                    head_name = c.name
                members.append(
                    ClusterMember(
                        concept_id=c.id,
                        name=c.name,
                        generality_score=generality[c.id],
                        coverage_score=float(c.coverage_score or 0.0),
                        is_gap=bool(c.is_gap),
                        gap_type=c.gap_type.value if c.gap_type else None,
                        is_cluster_head=is_head,
                    )
                )

            centroid = centroids[lbl]
            cohesion = float(np.mean([np.dot(emb[i], centroid) for i in indices]))
            avg_gen = float(
                np.mean([generality[concepts[i].id] for i in indices])
            )

            cluster_infos.append(
                ClusterInfo(
                    cluster_id=lbl,
                    label=head_name or f"Cluster {lbl}",
                    parent_concept_id=head_id,
                    parent_concept_name=head_name,
                    members=members,
                    size=len(indices),
                    avg_generality=round(avg_gen, 4),
                    cohesion=round(cohesion, 4),
                )
            )

        return ClusteringResult(
            project_id=project_id,
            total_concepts=n,
            clustered_concepts=n,
            noise_reassigned=noise_before,
            num_clusters=len(unique_labels),
            clusters=cluster_infos,
            bridge_relationships=bridges,
            algorithm_used=algorithm_used,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_clustering(
        self, emb: np.ndarray, min_cluster_size: int
    ) -> Tuple[List[int], str]:
        """
        Run HDBSCAN; fall back to KMeans when HDBSCAN yields < 2 real clusters.

        Embeddings are assumed to be L2-normalised; euclidean metric on
        unit vectors approximates cosine distance.
        """
        n = len(emb)

        # --- HDBSCAN --------------------------------------------------------
        try:
            import hdbscan as hdbscan_lib  # optional dependency

            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",          # L2-norm → ≈ cosine
                cluster_selection_method="eom",
                prediction_data=True,
            )
            raw_labels: np.ndarray = clusterer.fit_predict(emb)
            real_clusters = set(raw_labels.tolist()) - {-1}

            if len(real_clusters) >= 2:
                logger.info(
                    "HDBSCAN: %d clusters, %d noise points on %d concepts",
                    len(real_clusters),
                    int(np.sum(raw_labels == -1)),
                    n,
                )
                return raw_labels.tolist(), "hdbscan"

            logger.info(
                "HDBSCAN produced only %d real cluster(s); falling back to KMeans",
                len(real_clusters),
            )
        except Exception as exc:
            logger.warning("HDBSCAN failed (%s); falling back to KMeans", exc)

        # --- KMeans fallback ------------------------------------------------
        try:
            from sklearn.cluster import KMeans

            k = max(2, min(n // max(min_cluster_size, 2), 10))
            k = min(k, n)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(emb).tolist()
            logger.info("KMeans: %d clusters on %d concepts", k, n)
            return labels, "kmeans"
        except Exception as exc:
            logger.warning("KMeans failed (%s); assigning all to cluster 0", exc)

        return [0] * n, "none"

    def _reassign_noise(self, labels: List[int], emb: np.ndarray) -> List[int]:
        """Reassign noise points (label == -1) to the nearest cluster centroid."""
        labels = list(labels)
        noise_idx = [i for i, l in enumerate(labels) if l == -1]
        if not noise_idx:
            return labels

        real_labels = sorted({l for l in labels if l != -1})
        if not real_labels:
            return [0] * len(labels)

        # Centroids of real clusters
        centroids: Dict[int, np.ndarray] = {}
        label_arr = np.array(labels, dtype=float)
        for lbl in real_labels:
            mask = np.array([l == lbl for l in labels])
            centroids[lbl] = emb[mask].mean(axis=0)

        for i in noise_idx:
            best = max(
                real_labels,
                key=lambda lbl: float(np.dot(emb[i], centroids[lbl])),
            )
            labels[i] = best

        return labels

    def _detect_bridges(
        self,
        concepts: List[Any],
        emb: np.ndarray,
        labels: List[int],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Identify high-similarity concept pairs that span different clusters.

        Uses the full O(n²) dot-product matrix; skip for very large n.
        """
        n = len(concepts)
        if n > 500:
            logger.warning(
                "Skipping bridge detection: %d concepts exceeds limit of 500", n
            )
            return []

        # Full pairwise cosine similarities (unit vecs → dot product)
        sim_matrix = emb @ emb.T
        bridges: List[Dict[str, Any]] = []

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    bridges.append(
                        {
                            "source_concept_id": concepts[i].id,
                            "source_concept_name": concepts[i].name,
                            "source_cluster": int(labels[i]),
                            "target_concept_id": concepts[j].id,
                            "target_concept_name": concepts[j].name,
                            "target_cluster": int(labels[j]),
                            "similarity": round(sim, 4),
                        }
                    )

        return bridges

    # ------------------------------------------------------------------
    # Backward-compatible shim methods
    # ------------------------------------------------------------------

    async def cluster_concepts(
        self,
        embeddings: List[List[float]],
        concept_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Legacy method — accepts raw embeddings.

        Kept so that any external callers using the old ``ClusteringService``
        API continue to work without modification.
        """
        if not embeddings:
            return {"labels": [], "probabilities": [], "cluster_info": {}}

        emb = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms

        min_cs = max(self.min_cluster_size, len(emb) // 10)
        min_cs = max(2, min_cs)
        labels, _ = self._run_clustering(emb, min_cs)
        labels = self._reassign_noise(labels, emb)

        cluster_info: Dict[int, Dict] = {}
        for lbl in set(labels):
            member_ids = [
                (concept_ids[i] if concept_ids else i)
                for i, l in enumerate(labels)
                if l == lbl
            ]
            cluster_info[lbl] = {"size": len(member_ids), "concept_ids": member_ids}

        return {
            "labels": labels,
            "probabilities": [1.0] * len(labels),
            "cluster_info": cluster_info,
        }

    async def detect_hierarchy(
        self,
        labels: List[int],
        embeddings: List[List[float]],
        generality_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Legacy method — reconstruct hierarchy from raw labels and generality scores.

        Kept for backward compatibility with old callers.
        """
        hierarchy: Dict[str, Any] = {
            "cluster_relationships": [],
            "concept_hierarchies": {},
        }
        if not embeddings:
            return hierarchy

        emb = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms

        clusters: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            if lbl != -1:
                clusters.setdefault(lbl, []).append(idx)

        centroids = {
            lbl: emb[np.array(idxs)].mean(axis=0) for lbl, idxs in clusters.items()
        }
        label_list = sorted(centroids)

        for a in label_list:
            for b in label_list:
                if a >= b:
                    continue
                sim = float(np.dot(centroids[a], centroids[b]))
                if sim > 0.5:
                    hierarchy["cluster_relationships"].append(
                        {"cluster1": a, "cluster2": b, "similarity": round(sim, 4)}
                    )

        if generality_scores:
            for lbl, idxs in clusters.items():
                if len(idxs) < 2:
                    continue
                sorted_idxs = sorted(
                    idxs,
                    key=lambda i: (
                        generality_scores[i] if i < len(generality_scores) else 0.0
                    ),
                    reverse=True,
                )
                hierarchy["concept_hierarchies"][lbl] = {
                    "parent_concept": sorted_idxs[0],
                    "children_concepts": sorted_idxs[1:],
                }

        return hierarchy


# Backward-compatible alias
ClusteringService = ConceptClusterer
