"""
Clustering service using HDBSCAN for concept organization.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import hdbscan
from sklearn.preprocessing import StandardScaler

from app.config import settings

logger = logging.getLogger(__name__)


class ClusteringService:
    """Handles clustering and hierarchy detection using HDBSCAN."""

    def __init__(
        self,
        min_cluster_size: int = None,
        min_samples: int = None,
        metric: str = None
    ):
        """
        Initialize clustering service.

        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            metric: Distance metric to use
        """
        self.min_cluster_size = min_cluster_size or settings.HDBSCAN_MIN_CLUSTER_SIZE
        self.min_samples = min_samples or settings.HDBSCAN_MIN_SAMPLES
        self.metric = metric or settings.HDBSCAN_METRIC

    async def cluster_concepts(
        self,
        embeddings: List[List[float]],
        concept_ids: List[int] = None
    ) -> Dict[str, Any]:
        """
        Cluster concepts based on their embeddings.

        Args:
            embeddings: List of embedding vectors
            concept_ids: Optional list of concept IDs corresponding to embeddings

        Returns:
            Dictionary with 'labels', 'probabilities', 'cluster_info'
        """
        if not embeddings or len(embeddings) < self.min_cluster_size:
            logger.warning(f"Not enough embeddings for clustering: {len(embeddings)}")
            return {
                "labels": [-1] * len(embeddings),
                "probabilities": [0.0] * len(embeddings),
                "cluster_info": {}
            }

        try:
            # Convert to numpy array
            X = np.array(embeddings)

            # Normalize embeddings
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_method='eom',  # Excess of Mass
                prediction_data=True
            )

            labels = clusterer.fit_predict(X_scaled)
            probabilities = clusterer.probabilities_

            # Get cluster information
            cluster_info = await self._analyze_clusters(
                labels,
                probabilities,
                embeddings,
                concept_ids
            )

            logger.info(
                f"Clustered {len(embeddings)} concepts into "
                f"{len(set(labels)) - (1 if -1 in labels else 0)} clusters "
                f"({sum(labels == -1)} noise points)"
            )

            return {
                "labels": labels.tolist(),
                "probabilities": probabilities.tolist(),
                "cluster_info": cluster_info,
                "condensed_tree": clusterer.condensed_tree_ if hasattr(clusterer, 'condensed_tree_') else None
            }

        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {
                "labels": [-1] * len(embeddings),
                "probabilities": [0.0] * len(embeddings),
                "cluster_info": {}
            }

    async def _analyze_clusters(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        embeddings: List[List[float]],
        concept_ids: List[int] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze cluster characteristics.

        Args:
            labels: Cluster labels
            probabilities: Membership probabilities
            embeddings: Original embeddings
            concept_ids: Optional concept IDs

        Returns:
            Dictionary mapping cluster_id to cluster info
        """
        cluster_info = {}
        unique_labels = set(labels)

        for cluster_id in unique_labels:
            if cluster_id == -1:  # Noise points
                continue

            # Get indices of concepts in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Calculate cluster stats
            cluster_embeddings = [embeddings[i] for i in cluster_indices]
            cluster_probs = probabilities[cluster_mask]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Calculate cohesion (average distance to centroid)
            distances = [
                np.linalg.norm(np.array(emb) - centroid)
                for emb in cluster_embeddings
            ]
            cohesion = 1.0 / (1.0 + np.mean(distances))  # Normalize to 0-1

            cluster_info[int(cluster_id)] = {
                "size": len(cluster_indices),
                "concept_ids": [concept_ids[i] for i in cluster_indices] if concept_ids else cluster_indices.tolist(),
                "avg_probability": float(np.mean(cluster_probs)),
                "cohesion": float(cohesion),
                "centroid": centroid.tolist()
            }

        return cluster_info

    async def detect_hierarchy(
        self,
        labels: List[int],
        embeddings: List[List[float]],
        generality_scores: List[float] = None
    ) -> Dict[str, Any]:
        """
        Detect hierarchical structure within and between clusters.

        Args:
            labels: Cluster labels
            embeddings: Embedding vectors
            generality_scores: Optional generality scores for concepts

        Returns:
            Dictionary with hierarchy information
        """
        try:
            hierarchy = {
                "cluster_relationships": [],
                "concept_hierarchies": {}
            }

            # Group by cluster
            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)

            # Find inter-cluster relationships
            cluster_centroids = {}
            for cluster_id, indices in clusters.items():
                if cluster_id == -1:  # Skip noise
                    continue

                cluster_embeddings = [embeddings[i] for i in indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                cluster_centroids[cluster_id] = centroid

            # Calculate cluster similarities
            for cluster1 in cluster_centroids:
                for cluster2 in cluster_centroids:
                    if cluster1 >= cluster2:
                        continue

                    similarity = self._cosine_similarity(
                        cluster_centroids[cluster1],
                        cluster_centroids[cluster2]
                    )

                    if similarity > 0.5:  # Threshold for related clusters
                        hierarchy["cluster_relationships"].append({
                            "cluster1": int(cluster1),
                            "cluster2": int(cluster2),
                            "similarity": float(similarity)
                        })

            # Detect hierarchies within clusters using generality
            if generality_scores:
                for cluster_id, indices in clusters.items():
                    if cluster_id == -1 or len(indices) < 2:
                        continue

                    # Sort by generality within cluster
                    cluster_concepts = [
                        (idx, generality_scores[idx]) for idx in indices
                    ]
                    cluster_concepts.sort(key=lambda x: x[1], reverse=True)

                    # Most general concept is potential parent
                    parent_idx = cluster_concepts[0][0]
                    children_indices = [idx for idx, _ in cluster_concepts[1:]]

                    hierarchy["concept_hierarchies"][int(cluster_id)] = {
                        "parent_concept": parent_idx,
                        "children_concepts": children_indices
                    }

            return hierarchy

        except Exception as e:
            logger.error(f"Error detecting hierarchy: {e}")
            return {"cluster_relationships": [], "concept_hierarchies": {}}

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def get_outliers(
        self,
        labels: List[int],
        probabilities: List[float],
        threshold: float = 0.3
    ) -> List[int]:
        """
        Identify outlier concepts (noise or low probability).

        Args:
            labels: Cluster labels
            probabilities: Membership probabilities
            threshold: Probability threshold for outliers

        Returns:
            List of outlier indices
        """
        outliers = []

        for idx, (label, prob) in enumerate(zip(labels, probabilities)):
            if label == -1 or prob < threshold:
                outliers.append(idx)

        logger.info(f"Found {len(outliers)} outlier concepts")
        return outliers
