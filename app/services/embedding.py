"""
Embedding generation service using Ollama API.
"""
import httpx
import logging
from typing import List, Optional
import asyncio

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles embedding generation via Ollama."""

    def __init__(self):
        """Initialize embedding service."""
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_EMBED_MODEL
        self.timeout = httpx.Timeout(60.0, connect=10.0)

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text.strip()
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding")

                    if embedding and len(embedding) == settings.VECTOR_DIMENSION:
                        return embedding
                    else:
                        logger.error(f"Unexpected embedding dimension: {len(embedding) if embedding else 0}")
                        return None
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process concurrently

        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            # Generate embeddings concurrently for batch
            tasks = [self.generate_embedding(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for j, emb in enumerate(batch_embeddings):
                if isinstance(emb, Exception):
                    logger.error(f"Batch embedding failed for text {i + j}: {emb}")
                    embeddings.append(None)
                else:
                    embeddings.append(emb)

        logger.info(f"Generated {sum(1 for e in embeddings if e is not None)}/{len(texts)} embeddings")
        return embeddings

    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        if not embedding1 or not embedding2:
            return 0.0

        if len(embedding1) != len(embedding2):
            logger.error("Embedding dimension mismatch")
            return 0.0

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def find_similar_texts(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[tuple[int, float]]:
        """
        Find most similar texts based on embeddings.

        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for idx, candidate in enumerate(candidate_embeddings):
            if candidate:
                similarity = await self.compute_similarity(query_embedding, candidate)
                if similarity >= threshold:
                    similarities.append((idx, similarity))

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def check_ollama_health(self) -> bool:
        """
        Check if Ollama service is available.

        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
