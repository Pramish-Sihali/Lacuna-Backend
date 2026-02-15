"""
Knowledge gap detection service.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

from app.config import settings
from app.utils.helpers import cosine_similarity

logger = logging.getLogger(__name__)


class GapDetector:
    """Handles detection of knowledge gaps in concept networks."""

    def __init__(
        self,
        similarity_threshold: float = None,
        coverage_threshold: float = None
    ):
        """
        Initialize gap detector.

        Args:
            similarity_threshold: Threshold for semantic similarity
            coverage_threshold: Minimum coverage for well-explored concepts
        """
        self.similarity_threshold = similarity_threshold or settings.GAP_SIMILARITY_THRESHOLD
        self.coverage_threshold = coverage_threshold or settings.MIN_COVERAGE_THRESHOLD

    async def detect_all_gaps(
        self,
        concepts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        claims: Dict[int, List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect all types of knowledge gaps.

        Args:
            concepts: List of concepts
            relationships: List of relationships
            claims: Optional claims data

        Returns:
            List of detected gaps with metadata
        """
        gaps = []

        # Detect different types of gaps
        missing_links = await self.detect_missing_links(concepts, relationships)
        under_explored = await self.detect_under_explored(concepts, claims)
        contradictory = await self.detect_contradictory(concepts, claims)
        isolated = await self.detect_isolated_concepts(concepts, relationships)

        gaps.extend(missing_links)
        gaps.extend(under_explored)
        gaps.extend(contradictory)
        gaps.extend(isolated)

        logger.info(f"Detected {len(gaps)} total knowledge gaps")
        return gaps

    async def detect_missing_links(
        self,
        concepts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect missing links between concepts that should be connected.

        Args:
            concepts: List of concepts
            relationships: Existing relationships

        Returns:
            List of missing link gaps
        """
        gaps = []

        # Build relationship lookup
        existing_rels = set()
        for rel in relationships:
            existing_rels.add((rel["source_concept_id"], rel["target_concept_id"]))
            existing_rels.add((rel["target_concept_id"], rel["source_concept_id"]))

        # Find concepts that are semantically similar but not connected
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i + 1:]:
                # Skip if already connected
                if (concept1["id"], concept2["id"]) in existing_rels:
                    continue

                # Check semantic similarity
                if concept1.get("embedding") and concept2.get("embedding"):
                    similarity = cosine_similarity(
                        concept1["embedding"],
                        concept2["embedding"]
                    )

                    # High similarity but no connection = missing link
                    if similarity > self.similarity_threshold:
                        gaps.append({
                            "gap_type": "missing_link",
                            "concept_ids": [concept1["id"], concept2["id"]],
                            "severity": similarity,
                            "description": f"Missing connection between '{concept1['name']}' and '{concept2['name']}'",
                            "metadata": {
                                "similarity": similarity,
                                "concept1_name": concept1["name"],
                                "concept2_name": concept2["name"]
                            }
                        })

        logger.info(f"Found {len(gaps)} missing link gaps")
        return gaps

    async def detect_under_explored(
        self,
        concepts: List[Dict[str, Any]],
        claims: Dict[int, List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect under-explored concepts with low coverage.

        Args:
            concepts: List of concepts
            claims: Optional claims data

        Returns:
            List of under-explored gaps
        """
        gaps = []

        for concept in concepts:
            coverage = concept.get("coverage_score", 0.0)
            concept_id = concept["id"]

            # Check coverage score
            if coverage < self.coverage_threshold:
                # Count supporting claims
                claim_count = len(claims.get(concept_id, [])) if claims else 0

                severity = 1.0 - coverage

                gaps.append({
                    "gap_type": "under_explored",
                    "concept_ids": [concept_id],
                    "severity": severity,
                    "description": f"Concept '{concept['name']}' is under-explored",
                    "metadata": {
                        "coverage_score": coverage,
                        "claim_count": claim_count,
                        "concept_name": concept["name"]
                    }
                })

        logger.info(f"Found {len(gaps)} under-explored concept gaps")
        return gaps

    async def detect_contradictory(
        self,
        concepts: List[Dict[str, Any]],
        claims: Dict[int, List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect concepts with contradictory information.

        Args:
            concepts: List of concepts
            claims: Optional claims data

        Returns:
            List of contradictory gaps
        """
        gaps = []

        if not claims:
            return gaps

        for concept in concepts:
            concept_id = concept["id"]
            concept_claims = claims.get(concept_id, [])

            # Count contradictory claims
            contradictory_count = sum(
                1 for claim in concept_claims
                if claim.get("claim_type") == "contradicts"
            )

            # Check consensus score
            consensus = concept.get("consensus_score", 1.0)

            if contradictory_count > 0 or consensus < 0.5:
                severity = max(
                    contradictory_count / max(len(concept_claims), 1),
                    1.0 - consensus
                )

                gaps.append({
                    "gap_type": "contradictory",
                    "concept_ids": [concept_id],
                    "severity": severity,
                    "description": f"Concept '{concept['name']}' has contradictory information",
                    "metadata": {
                        "contradictory_claims": contradictory_count,
                        "total_claims": len(concept_claims),
                        "consensus_score": consensus,
                        "concept_name": concept["name"]
                    }
                })

        logger.info(f"Found {len(gaps)} contradictory gaps")
        return gaps

    async def detect_isolated_concepts(
        self,
        concepts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect isolated concepts with no or few connections.

        Args:
            concepts: List of concepts
            relationships: List of relationships

        Returns:
            List of isolated concept gaps
        """
        gaps = []

        # Count connections for each concept
        connection_count = defaultdict(int)
        for rel in relationships:
            connection_count[rel["source_concept_id"]] += 1
            connection_count[rel["target_concept_id"]] += 1

        for concept in concepts:
            concept_id = concept["id"]
            connections = connection_count[concept_id]

            # Isolated if no connections or very few
            if connections <= 1:
                severity = 1.0 if connections == 0 else 0.5

                gaps.append({
                    "gap_type": "isolated_concept",
                    "concept_ids": [concept_id],
                    "severity": severity,
                    "description": f"Concept '{concept['name']}' is isolated with few connections",
                    "metadata": {
                        "connection_count": connections,
                        "concept_name": concept["name"]
                    }
                })

        logger.info(f"Found {len(gaps)} isolated concept gaps")
        return gaps

    async def score_gaps(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score and prioritize gaps by importance.

        Args:
            gaps: List of gap dictionaries

        Returns:
            Gaps sorted by priority score
        """
        # Weight different gap types
        type_weights = {
            "contradictory": 1.0,  # Highest priority
            "missing_link": 0.9,
            "under_explored": 0.7,
            "isolated_concept": 0.6
        }

        for gap in gaps:
            gap_type = gap["gap_type"]
            severity = gap.get("severity", 0.5)
            type_weight = type_weights.get(gap_type, 0.5)

            # Priority score combines severity and type importance
            priority = severity * type_weight
            gap["priority_score"] = priority

        # Sort by priority
        gaps.sort(key=lambda x: x.get("priority_score", 0), reverse=True)

        return gaps

    async def suggest_research_directions(
        self,
        gaps: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Suggest research directions based on detected gaps.

        Args:
            gaps: List of detected gaps
            concepts: List of concepts

        Returns:
            List of research direction suggestions
        """
        suggestions = []
        concept_map = {c["id"]: c for c in concepts}

        # Group gaps by type
        gaps_by_type = defaultdict(list)
        for gap in gaps:
            gaps_by_type[gap["gap_type"]].append(gap)

        # Generate suggestions for missing links
        for gap in gaps_by_type.get("missing_link", [])[:5]:  # Top 5
            concept_ids = gap["concept_ids"]
            concept_names = [concept_map[cid]["name"] for cid in concept_ids]

            suggestions.append({
                "type": "explore_connection",
                "priority": gap.get("priority_score", 0.5),
                "description": f"Investigate the relationship between {' and '.join(concept_names)}",
                "related_concepts": concept_ids
            })

        # Generate suggestions for under-explored concepts
        for gap in gaps_by_type.get("under_explored", [])[:5]:  # Top 5
            concept_id = gap["concept_ids"][0]
            concept_name = concept_map[concept_id]["name"]

            suggestions.append({
                "type": "deep_dive",
                "priority": gap.get("priority_score", 0.5),
                "description": f"Conduct deeper research on '{concept_name}'",
                "related_concepts": [concept_id]
            })

        # Generate suggestions for contradictory areas
        for gap in gaps_by_type.get("contradictory", [])[:3]:  # Top 3
            concept_id = gap["concept_ids"][0]
            concept_name = concept_map[concept_id]["name"]

            suggestions.append({
                "type": "resolve_contradiction",
                "priority": gap.get("priority_score", 0.5),
                "description": f"Resolve contradictory information about '{concept_name}'",
                "related_concepts": [concept_id]
            })

        # Sort by priority
        suggestions.sort(key=lambda x: x["priority"], reverse=True)

        logger.info(f"Generated {len(suggestions)} research direction suggestions")
        return suggestions
