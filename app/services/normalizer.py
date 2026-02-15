"""
Concept normalization and deduplication service.
"""
import logging
from typing import List, Dict, Any, Set
from collections import defaultdict

from app.utils.helpers import clean_concept_name, cosine_similarity

logger = logging.getLogger(__name__)


class ConceptNormalizer:
    """Handles concept normalization and deduplication."""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize normalizer.

        Args:
            similarity_threshold: Minimum similarity to consider concepts as duplicates
        """
        self.similarity_threshold = similarity_threshold

    async def normalize_concepts(
        self,
        concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize and deduplicate a list of concepts.

        Args:
            concepts: List of concept dictionaries with 'name', 'description', etc.

        Returns:
            Deduplicated list of concepts with merged information
        """
        if not concepts:
            return []

        # First pass: exact name matching (after normalization)
        name_to_concepts = defaultdict(list)
        for concept in concepts:
            normalized_name = clean_concept_name(concept.get("name", ""))
            if normalized_name:
                name_to_concepts[normalized_name].append(concept)

        # Merge exact duplicates
        merged_concepts = []
        for normalized_name, concept_group in name_to_concepts.items():
            if len(concept_group) == 1:
                merged_concepts.append(concept_group[0])
            else:
                # Merge multiple concepts with same normalized name
                merged = await self._merge_duplicate_concepts(concept_group)
                merged_concepts.append(merged)

        logger.info(f"After exact matching: {len(concepts)} -> {len(merged_concepts)} concepts")

        # Second pass: embedding-based similarity (if embeddings available)
        final_concepts = await self._deduplicate_by_embedding(merged_concepts)

        logger.info(f"After embedding dedup: {len(merged_concepts)} -> {len(final_concepts)} concepts")

        return final_concepts

    async def _merge_duplicate_concepts(
        self,
        concepts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge multiple concept dictionaries into one.

        Args:
            concepts: List of duplicate concepts

        Returns:
            Merged concept dictionary
        """
        # Use the first concept as base
        merged = concepts[0].copy()

        # Merge descriptions (take longest or most informative)
        descriptions = [c.get("description", "") for c in concepts if c.get("description")]
        if descriptions:
            merged["description"] = max(descriptions, key=len)

        # Average numeric scores
        score_fields = ["generality_score", "coverage_score", "consensus_score"]
        for field in score_fields:
            scores = [c.get(field) for c in concepts if c.get(field) is not None]
            if scores:
                merged[field] = sum(scores) / len(scores)

        # Merge metadata
        all_metadata = {}
        for concept in concepts:
            if concept.get("metadata_json"):
                all_metadata.update(concept["metadata_json"])
        if all_metadata:
            merged["metadata_json"] = all_metadata

        return merged

    async def _deduplicate_by_embedding(
        self,
        concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate concepts using embedding similarity.

        Args:
            concepts: List of concepts with embeddings

        Returns:
            Deduplicated list
        """
        # Filter concepts that have embeddings
        concepts_with_embeddings = [
            c for c in concepts if c.get("embedding") is not None
        ]
        concepts_without_embeddings = [
            c for c in concepts if c.get("embedding") is None
        ]

        if not concepts_with_embeddings:
            return concepts

        # Track which concepts have been merged
        merged_indices: Set[int] = set()
        final_concepts = []

        for i, concept1 in enumerate(concepts_with_embeddings):
            if i in merged_indices:
                continue

            # Find all similar concepts
            similar_group = [concept1]
            merged_indices.add(i)

            for j, concept2 in enumerate(concepts_with_embeddings[i + 1:], start=i + 1):
                if j in merged_indices:
                    continue

                # Calculate similarity
                similarity = cosine_similarity(
                    concept1.get("embedding", []),
                    concept2.get("embedding", [])
                )

                if similarity >= self.similarity_threshold:
                    similar_group.append(concept2)
                    merged_indices.add(j)

            # Merge similar concepts
            if len(similar_group) > 1:
                merged = await self._merge_duplicate_concepts(similar_group)
                final_concepts.append(merged)
                logger.debug(f"Merged {len(similar_group)} similar concepts: {concept1.get('name')}")
            else:
                final_concepts.append(concept1)

        # Add concepts without embeddings
        final_concepts.extend(concepts_without_embeddings)

        return final_concepts

    async def find_canonical_concept(
        self,
        concept_name: str,
        existing_concepts: List[Dict[str, Any]],
        embedding: List[float] = None
    ) -> Dict[str, Any] | None:
        """
        Find if a concept already exists in canonical form.

        Args:
            concept_name: Name of concept to find
            existing_concepts: List of existing concepts
            embedding: Optional embedding for similarity matching

        Returns:
            Matching canonical concept or None
        """
        normalized_name = clean_concept_name(concept_name)

        # First check exact name match
        for concept in existing_concepts:
            if clean_concept_name(concept.get("name", "")) == normalized_name:
                return concept

        # If embedding provided, check similarity
        if embedding:
            for concept in existing_concepts:
                if concept.get("embedding"):
                    similarity = cosine_similarity(embedding, concept["embedding"])
                    if similarity >= self.similarity_threshold:
                        logger.debug(
                            f"Found similar concept: '{concept_name}' -> '{concept['name']}' "
                            f"(similarity: {similarity:.2f})"
                        )
                        return concept

        return None

    async def merge_concept_hierarchies(
        self,
        concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge concepts into hierarchical structure based on generality.

        Args:
            concepts: List of concepts with generality_score

        Returns:
            Concepts with parent_concept_id assigned where appropriate
        """
        # Sort by generality (most general first)
        sorted_concepts = sorted(
            concepts,
            key=lambda x: x.get("generality_score", 0.5),
            reverse=True
        )

        # Build hierarchy
        for i, specific_concept in enumerate(sorted_concepts):
            if specific_concept.get("parent_concept_id"):
                continue  # Already has parent

            # Look for potential parent (more general, similar concept)
            for general_concept in sorted_concepts[:i]:
                # Check if general enough and similar enough
                if (general_concept.get("generality_score", 0) >
                        specific_concept.get("generality_score", 0) + 0.2):

                    # Check similarity if embeddings available
                    if (specific_concept.get("embedding") and
                            general_concept.get("embedding")):
                        similarity = cosine_similarity(
                            specific_concept["embedding"],
                            general_concept["embedding"]
                        )

                        if similarity >= 0.6:  # Lower threshold for hierarchy
                            specific_concept["parent_concept_id"] = general_concept.get("id")
                            logger.debug(
                                f"Set parent: '{specific_concept['name']}' -> '{general_concept['name']}'"
                            )
                            break

        return sorted_concepts
