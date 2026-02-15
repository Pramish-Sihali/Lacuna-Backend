"""
Text chunking service for semantic segmentation.
"""
import re
import logging
from typing import List, Dict, Any

from app.config import settings
from app.utils.helpers import chunk_text_by_tokens

logger = logging.getLogger(__name__)


class ChunkingService:
    """Handles intelligent text chunking for embedding and processing."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize chunking service.

        Args:
            chunk_size: Size of chunks in tokens (default from settings)
            chunk_overlap: Overlap between chunks in tokens (default from settings)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    async def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk document text into smaller segments.

        Args:
            text: Full document text
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dictionaries with 'content', 'index', and 'metadata'
        """
        # First try semantic chunking by paragraphs/sections
        chunks = await self._semantic_chunk(text)

        # If chunks are too large, split them further
        final_chunks = []
        for chunk_text in chunks:
            if len(chunk_text.split()) > self.chunk_size:
                sub_chunks = chunk_text_by_tokens(
                    chunk_text,
                    chunk_size=self.chunk_size,
                    overlap=self.chunk_overlap
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk_text)

        # Create chunk objects with metadata
        chunk_objects = []
        for idx, content in enumerate(final_chunks):
            if content.strip():  # Skip empty chunks
                chunk_obj = {
                    "content": content.strip(),
                    "chunk_index": idx,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_size": len(content.split()),
                    }
                }
                chunk_objects.append(chunk_obj)

        logger.info(f"Created {len(chunk_objects)} chunks from document")
        return chunk_objects

    async def _semantic_chunk(self, text: str) -> List[str]:
        """
        Perform semantic chunking based on document structure.

        Args:
            text: Document text

        Returns:
            List of semantically meaningful chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para.split())

            # If adding this paragraph would exceed chunk size, start new chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Keep last paragraph for overlap
                current_chunk = [current_chunk[-1]] if self.chunk_overlap > 0 else []
                current_size = len(current_chunk[0].split()) if current_chunk else 0

            current_chunk.append(para)
            current_size += para_size

        # Add final chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    async def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences (useful for claim extraction).

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    async def get_context_window(
        self,
        chunks: List[str],
        target_index: int,
        window_size: int = 1
    ) -> str:
        """
        Get a chunk with surrounding context.

        Args:
            chunks: List of all chunks
            target_index: Index of target chunk
            window_size: Number of chunks before/after to include

        Returns:
            Combined text with context
        """
        start_idx = max(0, target_index - window_size)
        end_idx = min(len(chunks), target_index + window_size + 1)

        context_chunks = chunks[start_idx:end_idx]
        return "\n\n".join(context_chunks)
