"""
Embedding generation service using AWS Bedrock (Amazon Titan Embeddings V2).

Provides:
- BedrockEmbeddingService: concurrency-limited, retrying, caching, normalizing embedder
- embed_document_chunks: persist chunk vectors + document-level average to PostgreSQL
- find_similar_chunks: pgvector cosine-distance similarity search
- OllamaEmbeddingService / EmbeddingService aliases for backward compatibility
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database_models import Chunk, Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level embedding cache: sha256(text) → normalized vector
# ---------------------------------------------------------------------------
_embedding_cache: Dict[str, List[float]] = {}


# ---------------------------------------------------------------------------
# Pure vector helpers
# ---------------------------------------------------------------------------

def _hash_text(content: str) -> str:
    """SHA-256 digest of a text string — used as cache key."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize(vector: List[float]) -> List[float]:
    """Return a unit-length copy of *vector* (required for pgvector cosine ops)."""
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0.0:
        return vector
    return [x / magnitude for x in vector]


def _average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """Element-wise mean of equal-length vectors, then L2-normalized."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    avg = [
        sum(emb[i] for emb in embeddings) / len(embeddings)
        for i in range(dim)
    ]
    return _normalize(avg)


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class BedrockEmbeddingService:
    """
    Embedding generation via AWS Bedrock (Titan Embeddings V2) with
    production-quality safeguards:

    * Semaphore caps concurrent Bedrock calls (MAX_CONCURRENT = 3)
    * Exponential-backoff retries on connection / throttling errors (MAX_RETRIES = 3)
    * Unit-length normalization before storage (needed for cosine similarity)
    * In-process content-hash cache — identical text is embedded only once
    * pgvector similarity search via the <=> cosine-distance operator
    """

    MAX_CONCURRENT: int = 3
    MAX_RETRIES: int = 3

    def __init__(self) -> None:
        self.model = settings.AWS_BEDROCK_EMBED_MODEL
        self.expected_dim = settings.VECTOR_DIMENSION
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    async def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Embed a single text string.

        Returns a normalized (unit-length) vector, or ``None`` on permanent
        failure.  Results are cached by SHA-256 of the stripped input text.
        """
        if not text or not text.strip():
            logger.warning("embed_text: received empty/blank text — skipping")
            return None

        text = text.strip()
        key = _hash_text(text)
        if key in _embedding_cache:
            return _embedding_cache[key]

        embedding = await self._call_bedrock_with_retry(text)
        if embedding is not None:
            _embedding_cache[key] = embedding
        return embedding

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
    ) -> List[Optional[List[float]]]:
        """
        Embed a list of texts.

        Dispatches requests concurrently within each batch while respecting
        the semaphore.

        Returns a list of the same length as *texts*; failed items are ``None``.
        """
        results: List[Optional[List[float]]] = []
        total_batches = math.ceil(len(texts) / batch_size) if texts else 0

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            logger.info(
                "embed_batch: batch %d/%d — %d texts",
                batch_num,
                total_batches,
                len(batch),
            )

            gathered = await asyncio.gather(
                *[self.embed_text(t) for t in batch],
                return_exceptions=True,
            )

            for idx, res in enumerate(gathered):
                if isinstance(res, Exception):
                    logger.error(
                        "embed_batch: item %d raised: %s", batch_start + idx, res
                    )
                    results.append(None)
                else:
                    results.append(res)  # type: ignore[arg-type]

        successful = sum(1 for r in results if r is not None)
        logger.info(
            "embed_batch: %d/%d embeddings generated successfully",
            successful,
            len(texts),
        )
        return results

    async def embed_document_chunks(
        self,
        document_id: int,
        db: AsyncSession,
    ) -> Tuple[int, int]:
        """
        Embed every un-embedded chunk for *document_id*, persist the vectors,
        and write an averaged document-level embedding into the document's
        ``metadata_json`` field.

        **Small-doc fast path:** if the document's ``metadata_json`` has
        ``is_small_doc=True``, per-chunk embeddings are skipped entirely.
        Instead a single document-level embedding is generated from
        ``content_text`` and stored in ``metadata_json["document_embedding"]``.

        Returns ``(embedded_count, total_count)``.
        """
        # Load the document record to check the small-doc flag
        doc_result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = doc_result.scalar_one_or_none()

        chunk_result = await db.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        chunks = chunk_result.scalars().all()
        total = len(chunks)

        if total == 0:
            logger.warning(
                "embed_document_chunks: document %d has no chunks", document_id
            )
            return 0, 0

        # --- Small-doc fast path: skip per-chunk embedding ---
        meta: Dict[str, Any] = dict(document.metadata_json or {}) if document else {}
        if meta.get("is_small_doc") and document is not None:
            # Generate a single document-level embedding from full text
            doc_emb = await self.embed_text(document.content_text) if document.content_text else None
            if doc_emb is not None:
                meta["document_embedding"] = [float(x) for x in doc_emb]
                meta["embedding_model"] = self.model
            meta["embedding_skipped"] = True
            meta["embedded_chunks"] = 0
            meta["total_chunks"] = total
            document.metadata_json = meta
            await db.commit()
            logger.info(
                "embed_document_chunks: small doc %d — chunk embedding skipped, "
                "document-level embedding %s",
                document_id,
                "generated" if doc_emb else "failed",
            )
            return 0, total

        # --- Normal path: embed each chunk ---
        pending = [c for c in chunks if c.embedding is None]
        already_done = total - len(pending)
        logger.info(
            "Document %d: %d already embedded, %d pending",
            document_id,
            already_done,
            len(pending),
        )

        embedded_count = already_done
        for chunk in pending:
            emb = await self.embed_text(chunk.content)
            if emb is not None:
                chunk.embedding = emb
                embedded_count += 1
            else:
                logger.warning(
                    "embed_document_chunks: failed to embed chunk id=%d", chunk.id
                )

        # Build document-level average embedding from all now-embedded chunks
        all_embeddings = [c.embedding for c in chunks if c.embedding is not None]
        if all_embeddings and document is not None:
            doc_emb = _average_embeddings(all_embeddings)
            meta = dict(document.metadata_json or {})
            meta["document_embedding"] = [float(x) for x in doc_emb]
            meta["embedding_model"] = self.model
            meta["embedded_chunks"] = embedded_count
            meta["total_chunks"] = total
            document.metadata_json = meta

        await db.commit()
        logger.info(
            "embed_document_chunks: document %d — %d/%d chunks embedded",
            document_id,
            embedded_count,
            total,
        )
        return embedded_count, total

    async def find_similar_chunks(
        self,
        query_embedding: List[float],
        db: AsyncSession,
        top_k: int = 10,
        threshold: float = 0.7,
        document_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the *top_k* most similar chunks to *query_embedding* using
        pgvector's cosine-distance operator (``<=>``).

        *threshold* is a **similarity** floor (0–1).  Internally converted to
        a distance ceiling: ``distance_threshold = 1 - threshold``.

        Optionally restrict results to a single *document_id*.
        """
        # pgvector expects the literal string "[a,b,c,...]"
        embedding_str = "[" + ",".join(f"{v:.8f}" for v in query_embedding) + "]"
        distance_threshold = 1.0 - threshold

        doc_filter_sql = "AND c.document_id = :doc_id" if document_id else ""
        params: Dict[str, Any] = {
            "embedding": embedding_str,
            "distance_threshold": distance_threshold,
            "top_k": top_k,
        }
        if document_id is not None:
            params["doc_id"] = document_id

        sql = text(
            f"""
            SELECT
                c.id            AS chunk_id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.metadata_json,
                d.filename,
                d.file_type,
                1 - (c.embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              {doc_filter_sql}
              AND (c.embedding <=> CAST(:embedding AS vector)) <= :distance_threshold
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
            """
        )

        result = await db.execute(sql, params)
        rows = result.mappings().all()

        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata_json": row["metadata_json"],
                "filename": row["filename"],
                "file_type": row["file_type"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]

    async def check_bedrock_health(self) -> bool:
        """Return ``True`` if Bedrock is reachable and can generate embeddings."""
        try:
            def _check():
                body = json.dumps({
                    "inputText": "health check",
                    "dimensions": self.expected_dim,
                    "normalize": True,
                })
                resp = self._client.invoke_model(
                    modelId=self.model,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(resp["body"].read())
                return "embedding" in result

            return await asyncio.to_thread(_check)
        except Exception as exc:
            logger.error("Bedrock health check failed: %s", exc)
            return False

    # Keep old name for backward compat
    check_ollama_health = check_bedrock_health

    # ------------------------------------------------------------------
    # Backward-compatible aliases (used by health.py, concepts.py,
    # brain_service.py which were written against the old EmbeddingService)
    # ------------------------------------------------------------------

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Alias for :meth:`embed_text` — kept for backward compatibility."""
        return await self.embed_text(text)

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
    ) -> List[Optional[List[float]]]:
        """Alias for :meth:`embed_batch` — kept for backward compatibility."""
        return await self.embed_batch(texts, batch_size)

    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Cosine similarity between two vectors.  Pure computation, no I/O.
        Kept for backward compatibility with brain_service.py.
        """
        if not embedding1 or not embedding2:
            return 0.0
        if len(embedding1) != len(embedding2):
            logger.error("compute_similarity: dimension mismatch")
            return 0.0

        dot = sum(a * b for a, b in zip(embedding1, embedding2))
        mag1 = math.sqrt(sum(a * a for a in embedding1))
        mag2 = math.sqrt(sum(b * b for b in embedding2))
        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0
        return dot / (mag1 * mag2)

    async def find_similar_texts(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[int, float]]:
        """
        In-memory similarity search over a pre-loaded list of embeddings.
        Kept for backward compatibility; prefer :meth:`find_similar_chunks`
        for database-backed search.
        """
        similarities: List[Tuple[int, float]] = []
        for idx, candidate in enumerate(candidate_embeddings):
            if candidate:
                score = await self.compute_similarity(query_embedding, candidate)
                if score >= threshold:
                    similarities.append((idx, score))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_bedrock_with_retry(self, input_text: str) -> Optional[List[float]]:
        """
        Call Bedrock Titan Embeddings V2 with up to MAX_RETRIES attempts.
        Uses the semaphore to cap concurrency.  Exponential backoff on
        transient errors (connection failures, throttling).
        """
        async with self._semaphore:
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    t0 = time.perf_counter()

                    def _invoke():
                        body = json.dumps({
                            "inputText": input_text,
                            "dimensions": self.expected_dim,
                            "normalize": True,
                        })
                        resp = self._client.invoke_model(
                            modelId=self.model,
                            body=body,
                            contentType="application/json",
                            accept="application/json",
                        )
                        return json.loads(resp["body"].read())

                    result = await asyncio.to_thread(_invoke)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    raw: Optional[List[float]] = result.get("embedding")
                    if not raw:
                        logger.error(
                            "Bedrock response missing 'embedding' field "
                            "(attempt %d/%d)",
                            attempt,
                            self.MAX_RETRIES,
                        )
                        if attempt < self.MAX_RETRIES:
                            await asyncio.sleep(2 ** (attempt - 1))
                        continue

                    if len(raw) != self.expected_dim:
                        logger.error(
                            "Dimension mismatch: expected %d, got %d",
                            self.expected_dim,
                            len(raw),
                        )
                        return None

                    # Titan V2 with normalize=True returns unit vectors,
                    # but we normalize anyway for safety
                    normalized = _normalize(raw)
                    logger.debug(
                        "Embedded %d chars → %d-dim in %.1f ms",
                        len(input_text),
                        self.expected_dim,
                        elapsed_ms,
                    )
                    return normalized

                except EndpointConnectionError as exc:
                    logger.warning(
                        "Bedrock connect error (attempt %d/%d): %s",
                        attempt,
                        self.MAX_RETRIES,
                        exc,
                    )
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(2 ** (attempt - 1))

                except ClientError as exc:
                    error_code = exc.response["Error"]["Code"]
                    if error_code == "ThrottlingException":
                        logger.warning(
                            "Bedrock throttled (attempt %d/%d): %s",
                            attempt,
                            self.MAX_RETRIES,
                            exc,
                        )
                        if attempt < self.MAX_RETRIES:
                            await asyncio.sleep(2 ** (attempt - 1))
                    else:
                        logger.error("Bedrock client error: %s", exc)
                        return None

                except Exception as exc:
                    logger.error("Unexpected error calling Bedrock: %s", exc)
                    return None

        logger.error(
            "All %d embedding attempts failed for text (length=%d)",
            self.MAX_RETRIES,
            len(input_text),
        )
        return None


# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly when you don't need DI
# ---------------------------------------------------------------------------
embedding_service = BedrockEmbeddingService()

# ---------------------------------------------------------------------------
# Backward-compatibility aliases — existing code imports these names
# ---------------------------------------------------------------------------
OllamaEmbeddingService = BedrockEmbeddingService
EmbeddingService = BedrockEmbeddingService
