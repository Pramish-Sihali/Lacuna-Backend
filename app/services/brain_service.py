"""
Central brain service for RAG and consensus building.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.services.embedding import EmbeddingService
from app.services.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)


class BrainService:
    """Central brain service for querying and consensus building."""

    def __init__(self):
        """Initialize brain service."""
        self.embedding_service = EmbeddingService()
        self.llm_extractor = LLMExtractor()

    async def query(
        self,
        query_text: str,
        chunks: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]],
        top_k: int = 5,
        include_gaps: bool = True
    ) -> Dict[str, Any]:
        """
        Query the knowledge base using RAG.

        Args:
            query_text: User query
            chunks: List of document chunks with embeddings
            concepts: List of concepts with embeddings
            top_k: Number of top results to retrieve
            include_gaps: Whether to include gap information

        Returns:
            Query response with answer and relevant information
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query_text)

        if not query_embedding:
            return {
                "answer": "Error generating query embedding",
                "relevant_concepts": [],
                "relevant_chunks": [],
                "confidence": 0.0
            }

        # Find relevant chunks
        relevant_chunks = await self._find_relevant_chunks(
            query_embedding,
            chunks,
            top_k=top_k
        )

        # Find relevant concepts
        relevant_concepts = await self._find_relevant_concepts(
            query_embedding,
            concepts,
            top_k=top_k
        )

        # Build context from chunks and concepts
        context = await self._build_context(relevant_chunks, relevant_concepts)

        # Generate answer using LLM
        answer = await self._generate_answer(query_text, context, include_gaps)

        # Calculate confidence based on relevance scores
        confidence = await self._calculate_confidence(relevant_chunks, relevant_concepts)

        return {
            "query": query_text,
            "answer": answer,
            "relevant_concepts": relevant_concepts,
            "relevant_chunks": [
                {
                    "content": chunk["content"],
                    "similarity": chunk["similarity"],
                    "document_id": chunk.get("document_id")
                }
                for chunk in relevant_chunks
            ],
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def build_consensus(
        self,
        concepts: List[Dict[str, Any]],
        claims: Dict[int, List[Dict[str, Any]]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build consensus view of the knowledge base.

        Args:
            concepts: List of all concepts
            claims: Mapping of concept_id to claims
            relationships: List of relationships

        Returns:
            Consensus summary and key findings
        """
        # Calculate consensus scores for concepts
        concepts_with_consensus = await self._calculate_consensus_scores(
            concepts,
            claims
        )

        # Identify high-consensus concepts
        high_consensus = [
            c for c in concepts_with_consensus
            if c.get("consensus_score", 0) > 0.7
        ]

        # Identify areas of disagreement
        low_consensus = [
            c for c in concepts_with_consensus
            if c.get("consensus_score", 0) < 0.5
        ]

        # Generate overall summary
        summary = await self._generate_consensus_summary(
            high_consensus,
            low_consensus,
            relationships
        )

        return {
            "summary": summary,
            "high_consensus_concepts": [
                {"id": c["id"], "name": c["name"], "score": c.get("consensus_score")}
                for c in high_consensus[:10]
            ],
            "areas_of_disagreement": [
                {"id": c["id"], "name": c["name"], "score": c.get("consensus_score")}
                for c in low_consensus[:10]
            ],
            "total_concepts": len(concepts),
            "avg_consensus": sum(
                c.get("consensus_score", 0.5) for c in concepts_with_consensus
            ) / len(concepts_with_consensus) if concepts_with_consensus else 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _find_relevant_chunks(
        self,
        query_embedding: List[float],
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find most relevant chunks for query."""
        chunk_similarities = []

        for chunk in chunks:
            if chunk.get("embedding"):
                similarity = await self.embedding_service.compute_similarity(
                    query_embedding,
                    chunk["embedding"]
                )
                chunk_similarities.append({
                    **chunk,
                    "similarity": similarity
                })

        # Sort by similarity and return top k
        chunk_similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return chunk_similarities[:top_k]

    async def _find_relevant_concepts(
        self,
        query_embedding: List[float],
        concepts: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find most relevant concepts for query."""
        concept_similarities = []

        for concept in concepts:
            if concept.get("embedding"):
                similarity = await self.embedding_service.compute_similarity(
                    query_embedding,
                    concept["embedding"]
                )
                concept_similarities.append({
                    **concept,
                    "similarity": similarity
                })

        # Sort by similarity and return top k
        concept_similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return concept_similarities[:top_k]

    async def _build_context(
        self,
        chunks: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]]
    ) -> str:
        """Build context string from chunks and concepts."""
        context_parts = ["# Relevant Information\n"]

        # Add concept information
        if concepts:
            context_parts.append("\n## Key Concepts:")
            for concept in concepts:
                context_parts.append(
                    f"\n- {concept['name']}: {concept.get('description', 'No description')}"
                )

        # Add chunk content
        if chunks:
            context_parts.append("\n\n## Relevant Text Excerpts:")
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(f"\n{i}. {chunk['content'][:500]}")

        return "\n".join(context_parts)

    async def _generate_answer(
        self,
        query: str,
        context: str,
        include_gaps: bool = True
    ) -> str:
        """Generate answer using LLM with context."""
        prompt = f"""Based on the following context, answer this question: {query}

{context}

Provide a comprehensive answer based on the information above. If the information is insufficient, acknowledge what is known and what gaps exist.

Answer:"""

        answer = await self.llm_extractor._call_llm(prompt, max_tokens=500)
        return answer.strip() if answer else "Unable to generate answer at this time."

    async def _calculate_confidence(
        self,
        chunks: List[Dict[str, Any]],
        concepts: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for query response."""
        if not chunks and not concepts:
            return 0.0

        # Average similarity scores
        chunk_scores = [c.get("similarity", 0) for c in chunks]
        concept_scores = [c.get("similarity", 0) for c in concepts]

        all_scores = chunk_scores + concept_scores
        avg_similarity = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Adjust confidence based on number of sources
        source_factor = min(len(chunks) + len(concepts), 10) / 10

        confidence = avg_similarity * 0.7 + source_factor * 0.3

        return min(confidence, 1.0)

    async def _calculate_consensus_scores(
        self,
        concepts: List[Dict[str, Any]],
        claims: Dict[int, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Calculate consensus scores for concepts based on claims."""
        concepts_with_scores = []

        for concept in concepts:
            concept_id = concept["id"]
            concept_claims = claims.get(concept_id, [])

            if not concept_claims:
                # No claims = neutral consensus
                consensus_score = 0.5
            else:
                # Calculate based on claim types
                supports = sum(1 for c in concept_claims if c.get("claim_type") == "supports")
                contradicts = sum(1 for c in concept_claims if c.get("claim_type") == "contradicts")
                total = len(concept_claims)

                # Higher support ratio = higher consensus
                consensus_score = (supports - contradicts * 0.5) / total if total > 0 else 0.5
                consensus_score = max(0.0, min(1.0, consensus_score))

            concepts_with_scores.append({
                **concept,
                "consensus_score": consensus_score
            })

        return concepts_with_scores

    async def _generate_consensus_summary(
        self,
        high_consensus: List[Dict[str, Any]],
        low_consensus: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Generate summary of consensus findings."""
        summary_parts = []

        # High consensus areas
        if high_consensus:
            concept_names = [c["name"] for c in high_consensus[:5]]
            summary_parts.append(
                f"Strong consensus exists on: {', '.join(concept_names)}"
            )

        # Areas of disagreement
        if low_consensus:
            concept_names = [c["name"] for c in low_consensus[:3]]
            summary_parts.append(
                f"Areas requiring further investigation: {', '.join(concept_names)}"
            )

        # Relationship insights
        if relationships:
            summary_parts.append(
                f"Knowledge base contains {len(relationships)} documented relationships between concepts"
            )

        return ". ".join(summary_parts) if summary_parts else "Insufficient data for consensus analysis."
