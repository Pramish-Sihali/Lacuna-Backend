"""
Text chunking service with section-aware semantic segmentation.

Splitting strategy (in priority order):
  1. Primary   — by detected document sections (from ParsedDocument.sections)
  2. Secondary — by paragraph boundaries when a section exceeds CHUNK_SIZE tokens
  3. Tertiary  — by sentence boundaries when a paragraph exceeds CHUNK_SIZE tokens
  4. Last-resort — hard word-boundary split for a single oversized sentence

Chunks within the same section receive CHUNK_OVERLAP-token overlap from the
tail of their predecessor.  Sentence boundaries are always respected.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from app.config import settings

if TYPE_CHECKING:
    from app.services.document_parser import ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting (whitespace approximation — no external dependency)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate token count via whitespace splitting (1 word ≈ 1 token)."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Text splitting helpers
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> List[str]:
    """Split on one or more blank lines."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


# Sentence boundary: end of sentence punctuation followed by whitespace and
# an uppercase letter (or closing quote/bracket before uppercase).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\(\[])")


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Never breaks inside a sentence — the regex only fires on clear
    end-of-sentence signals (. ! ?) followed by a capital letter.
    """
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _apply_overlap(chunks: List[str], overlap_tokens: int) -> List[str]:
    """
    Prepend the tail of the previous chunk (up to `overlap_tokens` words) to
    each subsequent chunk, creating context continuity across chunk boundaries.
    """
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        tail_words = prev_words[-overlap_tokens:] if len(prev_words) > overlap_tokens else prev_words
        result.append(" ".join(tail_words) + " " + chunks[i])
    return result


# ---------------------------------------------------------------------------
# ChunkingService
# ---------------------------------------------------------------------------

class ChunkingService:
    """
    Produces semantic chunks from a ParsedDocument or plain text string.

    When given a ParsedDocument the service uses the pre-detected section
    structure as the primary split boundary, then cascades down to paragraph
    and sentence boundaries as needed.

    When given a plain string it falls back to paragraph/sentence splitting
    without section metadata.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chunk_document(
        self,
        text_or_parsed: Union["ParsedDocument", str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document into overlapping text segments.

        Args:
            text_or_parsed: Either a ParsedDocument (preferred, enables
                            section-aware chunking) or a plain text string.
            metadata:       Extra key/value pairs added to every chunk's
                            metadata dict (e.g. {"filename": "paper.pdf"}).

        Returns:
            List of dicts with keys:
              - content              : str — the chunk text
              - chunk_index          : int — sequential index (0-based)
              - metadata             : dict — section_title, section_level,
                                       page_num, approximate_token_count,
                                       plus any caller-supplied metadata
        """
        # Defer import to break circular dependency (parser ↔ chunker)
        from app.services.document_parser import ParsedDocument  # noqa: PLC0415

        base_meta = metadata or {}

        if isinstance(text_or_parsed, ParsedDocument):
            return await self._chunk_parsed(text_or_parsed, base_meta)
        return await self._chunk_raw_text(str(text_or_parsed), base_meta)

    # ------------------------------------------------------------------
    # Section-aware path
    # ------------------------------------------------------------------

    async def _chunk_parsed(
        self,
        parsed: "ParsedDocument",
        base_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Chunk using section boundaries from a ParsedDocument."""
        all_chunks: List[Dict[str, Any]] = []

        for section in parsed.sections:
            body = section.content.strip()
            if not body:
                continue

            raw_pieces = self._split_section(body)
            raw_pieces = _apply_overlap(raw_pieces, self.chunk_overlap)

            for piece in raw_pieces:
                piece = piece.strip()
                if not piece:
                    continue
                all_chunks.append(
                    self._make_chunk(
                        content=piece,
                        index=len(all_chunks),
                        base_meta=base_meta,
                        section_title=section.title,
                        section_level=section.level,
                        page_num=section.page_num,
                    )
                )

        if not all_chunks:
            # Sections produced no usable content; fall back to full text
            logger.warning("No chunks from sections — falling back to full-text chunking")
            return await self._chunk_raw_text(parsed.full_text, base_meta)

        logger.info(
            f"Created {len(all_chunks)} chunks from {len(parsed.sections)} sections"
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Plain-text fallback path
    # ------------------------------------------------------------------

    async def _chunk_raw_text(
        self,
        text: str,
        base_meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Chunk plain text without section metadata."""
        raw_pieces = self._split_section(text)
        raw_pieces = _apply_overlap(raw_pieces, self.chunk_overlap)

        result: List[Dict[str, Any]] = []
        for piece in raw_pieces:
            piece = piece.strip()
            if not piece:
                continue
            result.append(
                self._make_chunk(
                    content=piece,
                    index=len(result),
                    base_meta=base_meta,
                    section_title="",
                    section_level=0,
                    page_num=0,
                )
            )

        logger.info(f"Created {len(result)} chunks from raw text")
        return result

    # ------------------------------------------------------------------
    # Core splitting logic
    # ------------------------------------------------------------------

    def _split_section(self, text: str) -> List[str]:
        """
        Recursively split `text` until every chunk is ≤ chunk_size tokens.

        Strategy:
          1. If text fits in one chunk → return as-is
          2. Split on paragraph boundaries; accumulate until next paragraph
             would overflow → emit current chunk, then continue
          3. Paragraphs that are themselves oversized → sentence-split them
          4. Sentences that are themselves oversized → hard word-boundary split
        """
        if _count_tokens(text) <= self.chunk_size:
            return [text]

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return [text]

        chunks: List[str] = []
        current_words: List[str] = []

        for para in paragraphs:
            if _count_tokens(para) > self.chunk_size:
                # Flush accumulated words first
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = []
                # Tertiary: split oversized paragraph on sentences
                chunks.extend(self._split_by_sentences(para))
                continue

            if (
                current_words
                and len(current_words) + _count_tokens(para) > self.chunk_size
            ):
                chunks.append(" ".join(current_words))
                current_words = []

            current_words.extend(para.split())

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks if chunks else [text]

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split `text` on sentence boundaries, accumulating until chunk_size.

        For individual sentences exceeding chunk_size a hard word-boundary
        split is used as a last resort to guarantee the size contract.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return [text]

        chunks: List[str] = []
        current_words: List[str] = []

        for sent in sentences:
            sent_tokens = _count_tokens(sent)

            if sent_tokens > self.chunk_size:
                # Sentence itself is too long — hard split on words
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = []
                words = sent.split()
                for i in range(0, len(words), self.chunk_size):
                    chunks.append(" ".join(words[i: i + self.chunk_size]))
                continue

            if (
                current_words
                and len(current_words) + sent_tokens > self.chunk_size
            ):
                chunks.append(" ".join(current_words))
                current_words = []

            current_words.extend(sent.split())

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks if chunks else [text]

    # ------------------------------------------------------------------
    # Chunk object factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk(
        content: str,
        index: int,
        base_meta: Dict[str, Any],
        section_title: str,
        section_level: int,
        page_num: int,
    ) -> Dict[str, Any]:
        return {
            "content": content,
            "chunk_index": index,
            "metadata": {
                **base_meta,
                "section_title": section_title,
                "section_level": section_level,
                "page_num": page_num,
                "approximate_token_count": _count_tokens(content),
            },
        }

    # ------------------------------------------------------------------
    # Utility methods (used by other services)
    # ------------------------------------------------------------------

    async def chunk_by_sentences(self, text: str) -> List[str]:
        """Return the text split into individual sentences."""
        return _split_sentences(text)

    async def get_context_window(
        self,
        chunks: List[str],
        target_index: int,
        window_size: int = 1,
    ) -> str:
        """
        Return the chunk at `target_index` surrounded by `window_size`
        neighbouring chunks on each side.
        """
        start = max(0, target_index - window_size)
        end = min(len(chunks), target_index + window_size + 1)
        return "\n\n".join(chunks[start:end])
