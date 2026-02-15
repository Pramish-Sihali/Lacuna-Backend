"""
Relationship detection and scoring service for concepts.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from app.utils.helpers import cosine_similarity
from app.services.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)


class RelationshipService:
    """Handles relationship detection and scoring between concepts."""

    def __init__(self):
        """Initialize relationship service."""
        self.llm_extractor = LLMExtractor()

    async def detect_relationships(
        self,
        concepts: List[Dict[str, Any]],
        claims: Dict[int, List[Dict[str, Any]]] = None,
        use_llm: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect relationships between concepts.

        Args:
            concepts: List of concept dictionaries with embeddings
            claims: Optional mapping of concept_id to claims
            use_llm: Whether to use LLM for relationship analysis

        Returns:
            List of relationship dictionaries
        """
        relationships = []

        # Build concept index
        concept_map = {c["id"]: c for c in concepts}

        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i + 1:]:
                # Calculate embedding similarity
                if concept1.get("embedding") and concept2.get("embedding"):
                    similarity = cosine_similarity(
                        concept1["embedding"],
                        concept2["embedding"]
                    )

                    # Only process if concepts are somewhat related
                    if similarity > 0.3:
                        relationship = await self._analyze_relationship(
                            concept1,
                            concept2,
                            similarity,
                            claims,
                            use_llm
                        )

                        if relationship:
                            relationships.append(relationship)

        logger.info(f"Detected {len(relationships)} relationships")
        return relationships

    async def _analyze_relationship(
        self,
        concept1: Dict[str, Any],
        concept2: Dict[str, Any],
        embedding_similarity: float,
        claims: Dict[int, List[Dict[str, Any]]] = None,
        use_llm: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze relationship between two specific concepts.

        Args:
            concept1: First concept
            concept2: Second concept
            embedding_similarity: Pre-computed embedding similarity
            claims: Optional claims data
            use_llm: Whether to use LLM analysis

        Returns:
            Relationship dictionary or None
        """
        # Determine relationship type based on heuristics
        rel_type, strength = await self._infer_relationship_type(
            concept1,
            concept2,
            embedding_similarity,
            claims
        )

        # Optionally use LLM for more accurate analysis
        confidence = 0.7
        evidence = {}

        if use_llm and rel_type != "similar":
            llm_result = await self.llm_extractor.analyze_relationships(
                concept1["name"],
                concept2["name"],
                context=f"{concept1.get('description', '')} {concept2.get('description', '')}"
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
            "evidence_json": evidence
        }

    async def _infer_relationship_type(
        self,
        concept1: Dict[str, Any],
        concept2: Dict[str, Any],
        similarity: float,
        claims: Dict[int, List[Dict[str, Any]]] = None
    ) -> Tuple[str, float]:
        """
        Infer relationship type using heuristics.

        Args:
            concept1: First concept
            concept2: Second concept
            similarity: Embedding similarity
            claims: Optional claims data

        Returns:
            Tuple of (relationship_type, strength)
        """
        gen1 = concept1.get("generality_score", 0.5)
        gen2 = concept2.get("generality_score", 0.5)

        # Check for parent-child based on generality difference
        gen_diff = abs(gen1 - gen2)
        if gen_diff > 0.3 and similarity > 0.6:
            return "parent_child", similarity

        # Check for prerequisite (one concept builds on another)
        # This would ideally use temporal or citation data
        if gen1 > gen2 + 0.2:
            return "prerequisite", similarity * 0.8

        # Check for contradictory claims
        if claims:
            if await self._has_contradictory_claims(
                concept1["id"],
                concept2["id"],
                claims
            ):
                return "contradicts", similarity * 0.9

        # Check for complementary (similar generality, high similarity)
        if gen_diff < 0.2 and similarity > 0.7:
            return "complements", similarity

        # Check for builds_on relationship
        if 0.5 < similarity < 0.7 and gen2 > gen1:
            return "builds_on", similarity

        # Default to similar
        return "similar", similarity

    async def _has_contradictory_claims(
        self,
        concept1_id: int,
        concept2_id: int,
        claims: Dict[int, List[Dict[str, Any]]]
    ) -> bool:
        """
        Check if two concepts have contradictory claims.

        Args:
            concept1_id: First concept ID
            concept2_id: Second concept ID
            claims: Claims data

        Returns:
            True if contradictory claims found
        """
        claims1 = claims.get(concept1_id, [])
        claims2 = claims.get(concept2_id, [])

        for claim1 in claims1:
            if claim1.get("claim_type") == "contradicts":
                for claim2 in claims2:
                    if claim2.get("claim_type") == "contradicts":
                        # Simple check - could be enhanced with embedding similarity
                        return True

        return False

    async def score_relationships(
        self,
        relationships: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score and rank relationships by importance.

        Args:
            relationships: List of relationships
            concepts: List of concepts

        Returns:
            Relationships with updated scores
        """
        concept_map = {c["id"]: c for c in concepts}

        for rel in relationships:
            source = concept_map.get(rel["source_concept_id"])
            target = concept_map.get(rel["target_concept_id"])

            if not source or not target:
                continue

            # Calculate importance score based on:
            # 1. Concept coverage scores
            # 2. Relationship strength
            # 3. Relationship type (some types are more important)

            type_weights = {
                "prerequisite": 1.0,
                "contradicts": 0.95,
                "builds_on": 0.9,
                "parent_child": 0.85,
                "complements": 0.7,
                "similar": 0.5
            }

            coverage_score = (
                source.get("coverage_score", 0.5) +
                target.get("coverage_score", 0.5)
            ) / 2

            type_weight = type_weights.get(rel["relationship_type"], 0.5)

            importance = (
                rel["strength"] * 0.4 +
                coverage_score * 0.3 +
                type_weight * 0.3
            )

            rel["importance_score"] = importance

        # Sort by importance
        relationships.sort(key=lambda x: x.get("importance_score", 0), reverse=True)

        return relationships

    async def build_relationship_graph(
        self,
        relationships: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build a graph representation of concept relationships.

        Args:
            relationships: List of relationships
            concepts: List of concepts

        Returns:
            Graph structure with nodes and edges
        """
        # Build adjacency list
        adjacency = {}
        for concept in concepts:
            adjacency[concept["id"]] = {
                "outgoing": [],
                "incoming": []
            }

        for rel in relationships:
            source_id = rel["source_concept_id"]
            target_id = rel["target_concept_id"]

            if source_id in adjacency:
                adjacency[source_id]["outgoing"].append({
                    "target": target_id,
                    "type": rel["relationship_type"],
                    "strength": rel["strength"]
                })

            if target_id in adjacency:
                adjacency[target_id]["incoming"].append({
                    "source": source_id,
                    "type": rel["relationship_type"],
                    "strength": rel["strength"]
                })

        # Calculate graph metrics
        metrics = {
            "total_concepts": len(concepts),
            "total_relationships": len(relationships),
            "avg_degree": sum(
                len(adj["outgoing"]) + len(adj["incoming"])
                for adj in adjacency.values()
            ) / len(adjacency) if adjacency else 0,
            "isolated_concepts": sum(
                1 for adj in adjacency.values()
                if len(adj["outgoing"]) == 0 and len(adj["incoming"]) == 0
            )
        }

        return {
            "adjacency": adjacency,
            "metrics": metrics
        }
