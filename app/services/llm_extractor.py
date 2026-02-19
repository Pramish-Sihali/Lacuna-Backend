"""
LLM-based extraction service for concepts, claims, and relationships.

Uses Ollama's /api/generate endpoint with qwen2.5:3b (or whatever OLLAMA_LLM_MODEL
is configured to).  All JSON prompts are stored as module-level constants so they
can be tuned without touching logic code.

Public API
----------
OllamaLLMService.extract_concepts(chunk_text, document_context) -> List[Dict]
OllamaLLMService.extract_claims(chunk_text, concepts)           -> List[Dict]
OllamaLLMService.extract_relationships(chunk_text, doc_title)   -> List[Dict]
OllamaLLMService.process_document(document_id, db)             -> ExtractionSummary

Backward compatibility
----------------------
LLMExtractor = OllamaLLMService (alias)
_call_llm, analyze_relationships, generate_summary kept for brain_service.py
extract_concepts(text) single-arg form kept for concepts.py router
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database_models import Chunk, Claim, ClaimType, Concept, Document
from app.utils.helpers import clean_concept_name, cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ExtractionSummary:
    """Returned by process_document to summarise what was extracted and saved."""

    document_id: int
    document_title: str
    chunks_processed: int
    chunks_skipped: int
    concepts_extracted: int   # raw count before dedup
    concepts_saved: int       # net new concepts added to DB
    claims_saved: int
    relationships_found: int
    errors: List[str]


# ---------------------------------------------------------------------------
# Prompt templates — edit these to tune LLM output without touching logic
# ---------------------------------------------------------------------------

_CONCEPT_PROMPT = """\
You are a research analyst extracting key concepts from academic text.

Given the following text from a document titled "{document_title}":

---
{chunk_text}
---

Extract the key concepts discussed in this text. For each concept provide:
1. concept_name: A clear, concise name (2-5 words)
2. description: A brief description of what this concept means in context (1-2 sentences)
3. specificity_level: Is this concept "broad" (e.g., "Machine Learning"), \
"medium" (e.g., "Convolutional Neural Networks"), \
or "specific" (e.g., "ResNet-50 skip connections")?
4. confidence: How confident are you this is a real, distinct concept? (0.0 to 1.0)

Extract between 3 and 8 concepts. Do not repeat the same concept twice.

Respond ONLY with valid JSON array. No explanation, no markdown:
[{{"concept_name": "...", "description": "...", "specificity_level": "broad|medium|specific", "confidence": 0.9}}]\
"""

_CONCEPT_RETRY_PROMPT = """\
Extract key concepts from this academic text as a JSON array.

Text:
{chunk_text}

Return ONLY a JSON array — nothing else, no markdown:
[{{"concept_name": "Example Concept", "description": "One or two sentences.", "specificity_level": "medium", "confidence": 0.8}}]\
"""

_CLAIM_PROMPT = """\
You are a research analyst extracting specific claims from academic text.

Given the following text:
---
{chunk_text}
---

The key concepts in this text are: {concepts_list}

Extract specific claims or assertions made in this text. For each claim provide:
1. claim_text: The specific claim stated in one clear sentence
2. related_concept: Which concept from the list above this claim relates to \
(use the exact concept name from the list)
3. claim_type: One of:
   "supports"     — affirms or provides evidence for the concept
   "contradicts"  — challenges or refutes the concept
   "extends"      — adds new scope or dimension to the concept
   "complements"  — provides an additional perspective on the concept
4. confidence: How explicitly is this claim stated? (0.0 = implied, 1.0 = explicitly stated)

Extract between 2 and 6 claims. Only include claims clearly present in the text.

Respond ONLY with valid JSON array. No explanation, no markdown:
[{{"claim_text": "...", "related_concept": "...", "claim_type": "supports|contradicts|extends|complements", "confidence": 0.9}}]\
"""

_CLAIM_RETRY_PROMPT = """\
Extract claims from this academic text as JSON.

Text:
{chunk_text}

Concepts: {concepts_list}

Return ONLY a JSON array — nothing else:
[{{"claim_text": "A specific assertion.", "related_concept": "Concept Name", "claim_type": "supports", "confidence": 0.8}}]\
"""

_RELATIONSHIP_PROMPT = """\
You are a research analyst identifying how text relates to other work.

Document title: "{document_title}"

Text:
---
{chunk_text}
---

Identify any places where this text explicitly references, supports, contradicts, or extends
other research, theories, or concepts. For each relationship provide:
1. source_concept: The concept or idea expressed in THIS text
2. target_concept: The external concept, theory, or prior work being referenced
3. relationship_type: One of "supports", "contradicts", "extends", "complements"
4. evidence_text: The exact sentence or phrase that reveals this relationship
5. confidence: How clear is this relationship? (0.0 to 1.0)

If no clear cross-reference relationships are found, return an empty array.

Respond ONLY with valid JSON array. No explanation, no markdown:
[{{"source_concept": "...", "target_concept": "...", "relationship_type": "...", "evidence_text": "...", "confidence": 0.8}}]\
"""

_RELATIONSHIP_RETRY_PROMPT = """\
Find cross-reference relationships in this text as JSON.

Text:
{chunk_text}

Return ONLY a JSON array (empty array [] if none found):
[{{"source_concept": "...", "target_concept": "...", "relationship_type": "supports", "evidence_text": "...", "confidence": 0.8}}]\
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_float_list(value: Any) -> List[float]:
    """Safely convert a pgvector / numpy / list value to List[float]."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class OllamaLLMService:
    """
    LLM extraction service via Ollama /api/generate.

    Limits concurrency to MAX_CONCURRENT simultaneous LLM calls.
    Retries JSON parsing up to MAX_JSON_RETRIES times with a simpler prompt.
    Handles qwen2.5:3b's tendency to wrap JSON in markdown code fences.
    """

    MAX_CONCURRENT: int = 2
    MAX_JSON_RETRIES: int = 2
    LLM_TIMEOUT: float = float(settings.OLLAMA_TIMEOUT)  # default 300s from config
    MIN_CHUNK_WORDS: int = 50    # chunks shorter than this are skipped

    SPECIFICITY_TO_GENERALITY: Dict[str, float] = {
        "broad": 0.8,
        "medium": 0.5,
        "specific": 0.2,
    }

    VALID_CLAIM_TYPES = frozenset({"supports", "contradicts", "extends", "complements"})
    VALID_SPECIFICITY = frozenset({"broad", "medium", "specific"})
    VALID_RELATIONSHIP_TYPES = frozenset({"supports", "contradicts", "extends", "complements"})

    # Expose templates as class attributes so callers can hot-patch them
    CONCEPT_PROMPT = _CONCEPT_PROMPT
    CONCEPT_RETRY_PROMPT = _CONCEPT_RETRY_PROMPT
    CLAIM_PROMPT = _CLAIM_PROMPT
    CLAIM_RETRY_PROMPT = _CLAIM_RETRY_PROMPT
    RELATIONSHIP_PROMPT = _RELATIONSHIP_PROMPT
    RELATIONSHIP_RETRY_PROMPT = _RELATIONSHIP_RETRY_PROMPT

    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_LLM_MODEL
        self.timeout = httpx.Timeout(self.LLM_TIMEOUT, connect=10.0)
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    async def extract_concepts(
        self,
        chunk_text: str,
        document_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Extract key concepts from a chunk of text.

        Returns a list of dicts with keys:
          name, concept_name, description, specificity_level,
          generality_score, confidence

        The ``name`` key equals ``concept_name`` for backward compatibility
        with the existing concepts.py router.

        Args:
            chunk_text: The text to analyse (≤ 3 000 chars used).
            document_context: Document title / abstract used as prompt context.
                              Optional — defaults to empty string.
        """
        if self._is_too_short(chunk_text):
            return []

        document_title = document_context.strip() or "Unknown Document"
        truncated = chunk_text[:3000]

        prompt = self.CONCEPT_PROMPT.format(
            document_title=document_title,
            chunk_text=truncated,
        )
        retry_prompt = self.CONCEPT_RETRY_PROMPT.format(chunk_text=truncated)

        success, raw = await self._call_llm_json(prompt, retry_prompt=retry_prompt)
        if not success:
            return []

        if not isinstance(raw, list):
            raw = [raw] if isinstance(raw, dict) else []

        seen: set = set()
        concepts: List[Dict[str, Any]] = []

        for item in raw:
            if not isinstance(item, dict):
                continue

            name = str(item.get("concept_name", "")).strip()
            if not name:
                continue

            norm = clean_concept_name(name)
            if norm in seen:          # within-chunk deduplication
                continue
            seen.add(norm)

            specificity = str(item.get("specificity_level", "medium")).lower()
            if specificity not in self.VALID_SPECIFICITY:
                specificity = "medium"

            concepts.append({
                "name": name,           # backward compat key
                "concept_name": name,
                "description": str(item.get("description", "")).strip(),
                "specificity_level": specificity,
                "generality_score": self.SPECIFICITY_TO_GENERALITY[specificity],
                "confidence": self._clamp(item.get("confidence", 0.7)),
            })

        logger.info("extract_concepts: %d concepts from chunk", len(concepts))
        return concepts

    async def extract_claims(
        self,
        chunk_text: str,
        concepts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Extract specific claims/assertions made in *chunk_text*.

        Args:
            chunk_text: Source text.
            concepts: List of concept names already extracted from this chunk.
                      Used to anchor each claim to a concept.

        Returns a list of dicts with keys:
          claim_text, related_concept, claim_type, confidence
        """
        if self._is_too_short(chunk_text) or not concepts:
            return []

        concepts_list = ", ".join(concepts[:10])
        truncated = chunk_text[:3000]

        prompt = self.CLAIM_PROMPT.format(
            chunk_text=truncated,
            concepts_list=concepts_list,
        )
        retry_prompt = self.CLAIM_RETRY_PROMPT.format(
            chunk_text=truncated,
            concepts_list=concepts_list,
        )

        success, raw = await self._call_llm_json(prompt, retry_prompt=retry_prompt)
        if not success:
            return []

        if not isinstance(raw, list):
            raw = [raw] if isinstance(raw, dict) else []

        claims: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            claim_text = str(item.get("claim_text", "")).strip()
            if not claim_text:
                continue

            claim_type = str(item.get("claim_type", "supports")).lower().strip()
            if claim_type not in self.VALID_CLAIM_TYPES:
                claim_type = "supports"

            claims.append({
                "claim_text": claim_text,
                "related_concept": str(item.get("related_concept", "")).strip(),
                "claim_type": claim_type,
                "confidence": self._clamp(item.get("confidence", 0.7)),
            })

        logger.info("extract_claims: %d claims from chunk", len(claims))
        return claims

    async def extract_relationships(
        self,
        chunk_text: str,
        document_title: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract cross-reference relationships mentioned in *chunk_text*.

        Returns a list of dicts with keys:
          source_concept, target_concept, relationship_type,
          evidence_text, confidence
        """
        if self._is_too_short(chunk_text):
            return []

        truncated = chunk_text[:3000]
        prompt = self.RELATIONSHIP_PROMPT.format(
            document_title=document_title or "Unknown Document",
            chunk_text=truncated,
        )
        retry_prompt = self.RELATIONSHIP_RETRY_PROMPT.format(chunk_text=truncated)

        success, raw = await self._call_llm_json(prompt, retry_prompt=retry_prompt)
        if not success:
            return []

        if not isinstance(raw, list):
            return []

        relationships: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue

            source = str(item.get("source_concept", "")).strip()
            target = str(item.get("target_concept", "")).strip()
            if not source or not target:
                continue

            rel_type = str(item.get("relationship_type", "complements")).lower()
            if rel_type not in self.VALID_RELATIONSHIP_TYPES:
                rel_type = "complements"

            relationships.append({
                "source_concept": source,
                "target_concept": target,
                "relationship_type": rel_type,
                "evidence_text": str(item.get("evidence_text", "")).strip(),
                "confidence": self._clamp(item.get("confidence", 0.7)),
            })

        logger.info(
            "extract_relationships: %d relationships from chunk", len(relationships)
        )
        return relationships

    # ------------------------------------------------------------------
    # Full document extraction pipeline
    # ------------------------------------------------------------------

    async def process_document(
        self,
        document_id: int,
        db: AsyncSession,
    ) -> ExtractionSummary:
        """
        Run the full extraction pipeline for a document:

        1. Load all chunks for the document.
        2. For each non-trivial chunk:
           a. extract_concepts
           b. extract_claims (anchored to extracted concepts)
           c. extract_relationships
        3. Deduplicate concepts:
           - exact name match (normalised) against existing project concepts
           - embedding cosine similarity ≥ 0.85 against existing embeddings
        4. Persist new Concept rows; embed each new concept.
        5. Persist Claim rows (linking document ↔ concept).
        6. Return ExtractionSummary.

        Note: concepts are project-scoped, not document-scoped.  Concepts
        already in the DB (from other documents) will be reused.
        """
        # Import here to avoid circular imports at module load time
        from app.services.embedding import OllamaEmbeddingService

        # --- load document ---
        doc_result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = doc_result.scalar_one_or_none()
        if document is None:
            raise ValueError(f"Document {document_id} not found in database")

        document_title = document.filename

        # --- load chunks ---
        chunk_result = await db.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        chunks = chunk_result.scalars().all()

        if not chunks:
            return ExtractionSummary(
                document_id=document_id,
                document_title=document_title,
                chunks_processed=0,
                chunks_skipped=0,
                concepts_extracted=0,
                concepts_saved=0,
                claims_saved=0,
                relationships_found=0,
                errors=["No chunks found for document"],
            )

        # --- load existing project concepts for deduplication ---
        existing_result = await db.execute(
            select(Concept).where(
                Concept.project_id == document.project_id
            )
        )
        existing_concepts = existing_result.scalars().all()
        # normalised_name → Concept ORM object
        existing_by_name: Dict[str, Concept] = {
            clean_concept_name(c.name): c
            for c in existing_concepts
            if c.name
        }

        embedding_svc = OllamaEmbeddingService()

        # --- per-chunk extraction ---
        chunks_processed = 0
        chunks_skipped = 0
        all_raw_concepts: List[Tuple[Dict, int]] = []   # (concept_dict, chunk_id)
        all_raw_claims: List[Tuple[Dict, int]] = []     # (claim_dict, chunk_id)
        errors: List[str] = []

        for chunk in chunks:
            if self._is_too_short(chunk.content):
                chunks_skipped += 1
                logger.debug(
                    "Skipping short chunk id=%d (~%d words)",
                    chunk.id,
                    len(chunk.content.split()),
                )
                continue

            try:
                concepts = await self.extract_concepts(
                    chunk.content, document_title
                )

                # Optimisation: skip claims when no concepts found (saves 1 LLM call)
                claims: List[Dict] = []
                if concepts:
                    concept_names = [c["concept_name"] for c in concepts]
                    claims = await self.extract_claims(chunk.content, concept_names)

                # Optimisation: skip per-chunk relationship extraction entirely.
                # The RelationshipDetector service does a much better job at
                # project level using cosine similarity + evidence + LLM.
                # This saves ~33% of all LLM calls during extraction.

                for c in concepts:
                    all_raw_concepts.append((c, chunk.id))
                for cl in claims:
                    all_raw_claims.append((cl, chunk.id))

                chunks_processed += 1
                logger.info(
                    "Chunk %d/%d (id=%d): %d concepts, %d claims",
                    chunks_processed,
                    len(chunks),
                    chunk.id,
                    len(concepts),
                    len(claims),
                )

            except Exception as exc:
                msg = f"Chunk id={chunk.id}: {exc}"
                errors.append(msg)
                logger.error("process_document error — %s", msg, exc_info=True)
                chunks_skipped += 1

        # --- deduplicate + save concepts ---
        #
        # batch_concept_map: normalised_name → Concept ORM object
        # Tracks both pre-existing concepts (reused) and newly created ones.
        batch_concept_map: Dict[str, Concept] = {}
        new_concepts_count = 0

        for concept_dict, _chunk_id in all_raw_concepts:
            concept_name: str = concept_dict["concept_name"]
            norm = clean_concept_name(concept_name)
            if not norm:
                continue

            # Already handled in this extraction run?
            if norm in batch_concept_map:
                continue

            # Exact name match against existing DB concepts?
            if norm in existing_by_name:
                batch_concept_map[norm] = existing_by_name[norm]
                continue

            # New concept — embed it so we can check embedding similarity
            embed_input = f"{concept_name} {concept_dict.get('description', '')}".strip()
            embedding = await embedding_svc.embed_text(embed_input)

            # Embedding-based deduplication vs. existing concepts
            # O(N) per new concept — acceptable for typical collection sizes.
            matched: Optional[Concept] = None
            if embedding:
                emb_list = list(embedding)
                for existing_c in existing_concepts:
                    existing_emb = _to_float_list(existing_c.embedding)
                    if not existing_emb:
                        continue
                    try:
                        sim = cosine_similarity(emb_list, existing_emb)
                        if sim >= 0.85:
                            matched = existing_c
                            logger.debug(
                                "Concept '%s' deduplicated → '%s' (cos=%.3f)",
                                concept_name,
                                existing_c.name,
                                sim,
                            )
                            break
                    except ValueError:
                        continue   # dimension mismatch — skip

            if matched is not None:
                batch_concept_map[norm] = matched
                existing_by_name[norm] = matched
                continue

            # Genuinely new — persist to DB
            new_concept = Concept(
                project_id=document.project_id,
                name=concept_name,
                description=concept_dict.get("description", ""),
                generality_score=concept_dict.get("generality_score", 0.5),
                coverage_score=0.5,
                consensus_score=1.0,
                embedding=embedding,
                metadata_json={
                    "specificity_level": concept_dict.get("specificity_level", "medium"),
                    "extraction_confidence": concept_dict.get("confidence", 0.7),
                    "source_document_id": document_id,
                },
            )
            db.add(new_concept)
            await db.flush()  # obtain auto-generated id immediately

            batch_concept_map[norm] = new_concept
            existing_by_name[norm] = new_concept
            existing_concepts.append(new_concept)  # include in future similarity checks
            new_concepts_count += 1

        # --- save claims ---
        claims_saved = 0
        for claim_dict, _chunk_id in all_raw_claims:
            related_name = claim_dict.get("related_concept", "")
            norm_related = clean_concept_name(related_name)

            # Resolve concept — exact match first
            concept_obj = batch_concept_map.get(norm_related)

            # Fuzzy fallback: substring containment
            if concept_obj is None and norm_related:
                for bname, bobj in batch_concept_map.items():
                    if norm_related in bname or bname in norm_related:
                        concept_obj = bobj
                        break

            if concept_obj is None:
                logger.debug(
                    "Cannot resolve concept '%s' for claim — skipping", related_name
                )
                continue

            # Map to DB enum (values are already validated as supports/contradicts/…)
            try:
                claim_type_enum = ClaimType(claim_dict["claim_type"])
            except (KeyError, ValueError):
                claim_type_enum = ClaimType.SUPPORTS

            db.add(
                Claim(
                    document_id=document_id,
                    concept_id=concept_obj.id,
                    claim_text=claim_dict["claim_text"],
                    claim_type=claim_type_enum,
                    confidence=claim_dict.get("confidence", 0.7),
                )
            )
            claims_saved += 1

        await db.commit()

        summary = ExtractionSummary(
            document_id=document_id,
            document_title=document_title,
            chunks_processed=chunks_processed,
            chunks_skipped=chunks_skipped,
            concepts_extracted=len(all_raw_concepts),
            concepts_saved=new_concepts_count,
            claims_saved=claims_saved,
            relationships_found=0,  # relationships detected at project level, not per-chunk
            errors=errors,
        )
        logger.info(
            "process_document %d: %d chunks processed, "
            "%d new concepts, %d claims",
            document_id,
            chunks_processed,
            new_concepts_count,
            claims_saved,
        )
        return summary

    # ------------------------------------------------------------------
    # Core LLM caller
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        POST to Ollama /api/generate and return the response text.

        Uses semaphore to cap concurrent LLM calls.  Returns empty string
        on any error (timeout, connection failure, non-200 response).
        """
        async with self._semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "num_predict": max_tokens,
                                "temperature": 0.1,  # low temp for deterministic JSON
                            },
                        },
                    )

                if resp.status_code == 200:
                    return resp.json().get("response", "")

                logger.error(
                    "_call_llm: Ollama returned HTTP %d: %s",
                    resp.status_code,
                    resp.text[:300],
                )
                return ""

            except httpx.TimeoutException:
                logger.error(
                    "_call_llm: request timed out after %.0f s", self.LLM_TIMEOUT
                )
                return ""
            except httpx.ConnectError as exc:
                logger.error("_call_llm: connection error — %s", exc)
                return ""
            except Exception as exc:
                logger.error("_call_llm: unexpected error — %s", exc)
                return ""

    async def _call_llm_json(
        self,
        prompt: str,
        max_tokens: int = 1500,
        retry_prompt: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """
        Call the LLM and attempt to parse the response as JSON.

        Retries up to MAX_JSON_RETRIES times.  On retry, uses *retry_prompt*
        (a simpler, more directive prompt) if provided, otherwise repeats the
        original prompt.

        Returns ``(success: bool, parsed_value: Any)``.
        """
        prompts = [prompt] + [retry_prompt or prompt] * (self.MAX_JSON_RETRIES - 1)

        for attempt, current_prompt in enumerate(prompts, start=1):
            response_text = await self._call_llm(current_prompt, max_tokens)

            if not response_text:
                # Empty response means timeout or connection error — retrying
                # the same call will likely time out again. Bail immediately.
                logger.warning(
                    "_call_llm_json: empty LLM response (attempt %d) — "
                    "skipping retries (likely timeout)",
                    attempt,
                )
                return False, []

            success, parsed = self._parse_json_robust(response_text)
            if success:
                if attempt > 1:
                    logger.info(
                        "_call_llm_json: JSON parsed successfully on attempt %d",
                        attempt,
                    )
                return True, parsed

            if attempt < self.MAX_JSON_RETRIES:
                logger.warning(
                    "_call_llm_json: JSON parse failed on attempt %d/%d, retrying",
                    attempt,
                    self.MAX_JSON_RETRIES,
                )

        logger.error(
            "_call_llm_json: all %d JSON parse attempts failed", self.MAX_JSON_RETRIES
        )
        return False, []

    # ------------------------------------------------------------------
    # Robust JSON parsing
    # ------------------------------------------------------------------

    def _parse_json_robust(self, response: str) -> Tuple[bool, Any]:
        """
        Try multiple strategies to parse JSON from potentially messy LLM output.

        Handles:
        - Markdown code fences (```json … ```, ``` … ```)
        - Trailing commas before ] or }
        - Python-style True / False / None
        - Surrounding prose — finds the innermost [...] or {...} block
        - Missing closing bracket (adds one and retries)

        Returns ``(success, parsed_value)``.
        """
        if not response:
            return False, []

        text = response.strip()

        # Strategy 1: direct parse
        ok, val = self._try_json(text)
        if ok:
            return True, val

        # Strategy 2: strip markdown code fences
        stripped = self._strip_code_fences(text)
        if stripped != text:
            ok, val = self._try_json(stripped)
            if ok:
                return True, val
            text = stripped  # work on stripped version from here

        # Strategy 3: fix common JSON mangling
        fixed = self._fix_json_issues(text)
        ok, val = self._try_json(fixed)
        if ok:
            return True, val

        # Strategy 4: extract JSON structure from surrounding prose
        for bracket_pair in (("[", "]"), ("{", "}")):
            fragment = self._extract_json_structure(text, *bracket_pair)
            if fragment:
                ok, val = self._try_json(fragment)
                if ok:
                    return True, val
                ok, val = self._try_json(self._fix_json_issues(fragment))
                if ok:
                    return True, val

        # Strategy 5: attempt to close a truncated array / object
        for suffix in ("]", "}", "}]"):
            ok, val = self._try_json(fixed + suffix)
            if ok:
                logger.debug("_parse_json_robust: recovered with suffix %r", suffix)
                return True, val

        logger.warning(
            "_parse_json_robust: all strategies failed. Preview: %s",
            response[:400],
        )
        return False, []

    @staticmethod
    def _try_json(text: str) -> Tuple[bool, Any]:
        try:
            return True, json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return False, None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```json / ``` delimiters that LLMs often wrap output in."""
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```(?:json|python|javascript|text)?\s*\n?", "", text, flags=re.IGNORECASE)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    @staticmethod
    def _fix_json_issues(text: str) -> str:
        """Repair the most common JSON mangling patterns from LLMs."""
        # Trailing commas before ] or }
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        # Python → JSON literals
        text = re.sub(r"\bTrue\b", "true", text)
        text = re.sub(r"\bFalse\b", "false", text)
        text = re.sub(r"\bNone\b", "null", text)
        # Strip inline comments (// …) — JSON doesn't allow them
        text = re.sub(r"//[^\n]*", "", text)
        return text.strip()

    @staticmethod
    def _extract_json_structure(text: str, open_b: str, close_b: str) -> str:
        """
        Find the first complete balanced open_b … close_b structure in *text*.
        Returns the matched fragment, or empty string if not found.
        """
        start = text.find(open_b)
        if start == -1:
            return ""

        depth = 0
        in_string = False
        escape_next = False

        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_b:
                depth += 1
            elif ch == close_b:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_too_short(self, text: str) -> bool:
        """Return True if text is too short for meaningful extraction."""
        return len(text.split()) < self.MIN_CHUNK_WORDS

    @staticmethod
    def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
        """Parse *value* as float, clamped to [lo, hi]; returns midpoint on error."""
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return (lo + hi) / 2.0

    # ------------------------------------------------------------------
    # Backward-compatible methods
    # (used by brain_service.py, concepts.py — do not remove)
    # ------------------------------------------------------------------

    async def analyze_relationships(
        self,
        concept1: str,
        concept2: str,
        context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse the relationship between two named concepts.
        Kept for backward compatibility with brain_service.py.
        """
        prompt = (
            f"Analyze the relationship between these two concepts:\n"
            f"Concept 1: {concept1}\n"
            f"Concept 2: {concept2}\n"
            f"{f'Context: {context[:1000]}' if context else ''}\n\n"
            "Determine:\n"
            "1. relationship_type: One of [prerequisite, builds_on, contradicts, "
            "complements, similar, parent_child]\n"
            "2. strength: 0-1\n"
            "3. confidence: 0-1\n"
            "4. explanation: brief explanation\n\n"
            "Return ONLY a JSON object:\n"
            '{"relationship_type": "...", "strength": 0.5, '
            '"confidence": 0.7, "explanation": "..."}'
        )

        response = await self._call_llm(prompt, max_tokens=300)
        if not response:
            return None

        _ok, parsed = self._parse_json_robust(response)
        if not isinstance(parsed, dict):
            return None

        valid_types = {
            "prerequisite", "builds_on", "contradicts",
            "complements", "similar", "parent_child",
        }
        rel_type = str(parsed.get("relationship_type", "similar")).lower()
        if rel_type not in valid_types:
            rel_type = "similar"

        return {
            "relationship_type": rel_type,
            "strength": self._clamp(parsed.get("strength", 0.5)),
            "confidence": self._clamp(parsed.get("confidence", 0.7)),
            "explanation": str(parsed.get("explanation", "")),
        }

    async def generate_summary(
        self,
        texts: List[str],
        max_length: int = 500,
    ) -> str:
        """
        Generate a free-text summary of multiple passages.
        Kept for backward compatibility.
        """
        combined = "\n\n".join(texts)[:5000]
        prompt = (
            f"Summarize the following research content in {max_length} words or less.\n"
            "Focus on key findings, concepts, and insights.\n\n"
            f"Content:\n{combined}\n\nSummary:"
        )
        result = await self._call_llm(prompt, max_tokens=max_length * 2)
        return result.strip() if result else ""


# ---------------------------------------------------------------------------
# Backward-compatibility alias — existing imports use `LLMExtractor`
# ---------------------------------------------------------------------------
LLMExtractor = OllamaLLMService
