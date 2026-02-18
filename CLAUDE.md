# Lacuna Backend - Project Context

## What is Lacuna?
AI-powered research discovery platform that helps academics visualize knowledge gaps in document collections. Users upload research papers (PDF/DOCX), the system extracts concepts, detects relationships, identifies knowledge gaps, and provides RAG querying.

## Tech Stack
- **Framework:** FastAPI 0.115 (async, Python 3.11+)
- **Server:** Uvicorn (ASGI)
- **Database:** PostgreSQL 16 + pgvector extension (vector similarity search)
- **ORM:** SQLAlchemy 2.0 (async) with Alembic migrations
- **LLM:** Ollama local inference (qwen2.5:3b for generation, nomic-embed-text for 768-dim embeddings)
- **Clustering:** HDBSCAN (density-based, no K required) with KMeans fallback
- **Doc Parsing:** PyMuPDF (PDF), python-docx (DOCX), Tesseract OCR (fallback)
- **Validation:** Pydantic v2 for request/response schemas
- **Frontend:** Next.js (separate repo)

## Project Structure
```
app/
├── main.py                 # FastAPI app entry, CORS, request-logging middleware, lifespan, routers
├── config.py               # Pydantic Settings from .env
├── database.py             # Async engine, session factory, init_db()
├── models/
│   ├── database_models.py  # 7 SQLAlchemy tables: projects, documents, chunks, concepts, claims, relationships, brain_state
│   └── schemas.py          # Pydantic request/response models (incl. ReactFlow map schemas)
├── routers/
│   ├── health.py           # GET /api/health (DB + Ollama status)
│   ├── documents.py        # Upload, embed, extract, process, list, detail, delete
│   ├── concepts.py         # extract, normalize, build, list, map (React Flow), cluster, relationships, detect-gaps, gaps, detail
│   ├── search.py           # GET /api/search/similar (vector similarity search)
│   ├── brain.py            # POST build, POST chat, GET status, POST rebuild
│   └── pipeline.py         # POST process-all, POST full-rebuild (orchestration endpoints)
├── services/
│   ├── document_parser.py  # PDF/DOCX text extraction with OCR fallback
│   ├── chunking.py         # Semantic chunking (500 tokens, 50 overlap)
│   ├── embedding.py        # OllamaEmbeddingService: embed, batch, similarity search, caching
│   ├── llm_extractor.py    # OllamaLLMService: concept/claim/relationship extraction, robust JSON parsing
│   ├── normalizer.py       # ConceptNormalizer: exact-name + AgglomerativeClustering dedup (>0.85 cosine)
│   ├── clustering.py       # ConceptClusterer: HDBSCAN+KMeans, generality scoring, 2-level hierarchy, bridge detection
│   ├── relationships.py    # RelationshipDetector: 5-step pipeline (similarity + evidence + LLM + scoring + dedup)
│   ├── gap_detector.py     # GapDetector: 3 gap passes (expected topics, bridging, weak coverage)
│   ├── brain_service.py    # BrainService: consensus scoring, brain state persistence, RAG chat
│   └── pipeline.py         # LacunaPipeline: master orchestrator (process_document, rebuild_project_knowledge, add_document_and_update)
└── utils/
    └── helpers.py          # cosine_similarity, text normalization, keyword extraction, hashing
```

## Database Schema (7 tables)
- **projects** - Research project containers (id, name, description)
- **documents** - Uploaded files + parsed text (FK: project_id)
- **chunks** - Text segments with 768-dim pgvector embeddings (FK: document_id)
- **concepts** - Extracted concepts with scores (generality, coverage, consensus), gap flags, cluster_label, parent_concept_id (FK: project_id)
- **claims** - Evidence linking documents to concepts with claim_type (supports|contradicts|extends|complements) (FK: document_id, concept_id)
- **relationships** - Concept-to-concept connections with type (prerequisite|builds_on|contradicts|complements|similar|parent_child), strength, confidence (FK: source/target concept)
- **brain_state** - Project-wide consensus tracking: summary_text, consensus_json (FK: project_id)

## Key Architecture Patterns
- **Async-first:** All DB and I/O operations use async/await
- **Dependency Injection:** `Depends(get_db)` provides DB sessions to endpoints
- **Service Layer:** Routers handle HTTP, services handle business logic
- **Pipeline Orchestrator:** `LacunaPipeline` in `services/pipeline.py` coordinates all services end-to-end; each stage is wrapped in try/except so partial failures don't abort the full run
- **Cascade Deletes:** Deleting a document cascades to chunks and claims
- **Batch Embedding:** Embeddings generated in batches of 10 with `asyncio.Semaphore(3)`
- **Per-document sessions:** Batch endpoints (`process-all`, `full-rebuild`, `embed-all`) open a fresh `AsyncSessionLocal()` per document so one failure doesn't roll back the entire batch
- **SQLAlchemy 2.0:** All raw SQL must be wrapped with `text()` — never pass bare strings to `conn.execute()`
- **`from __future__ import annotations`:** Present in service files for cleaner type hints. Do NOT use in router files that have `status_code=204` routes without explicit `response_model=None` (causes FastAPI assertion error)
- **204 DELETE routes:** Must include `response_model=None` explicitly: `@router.delete(..., status_code=204, response_model=None)`
- **Route ordering in concepts.py:** All static routes (e.g. `/build`, `/map`, `/gaps`, `/relationships`) must be registered BEFORE the dynamic `/{concept_id}` route to avoid path shadowing
- **Backward-compat aliases:** `EmbeddingService = OllamaEmbeddingService`, `LLMExtractor = OllamaLLMService`, `ClusteringService = ConceptClusterer`, `RelationshipService` (old class preserved at bottom of relationships.py)
- **L2 normalization:** All embedding vectors are unit-length before storage (required for pgvector cosine ops)
- **Re-runnable gap detection:** Synthetic gap concept nodes are tagged `metadata_json["is_synthetic_gap"] = True`; `_clear_old_gaps()` deletes them and resets flags on real concepts before each run
- **Request logging middleware:** Every non-health request is logged with method, path, status code, and elapsed ms; `X-Process-Time` header attached to all responses

## API Endpoints
```
GET  /api/health                        # System status (DB + Ollama + model availability)

POST /api/documents/upload              # Upload + parse + chunk
POST /api/documents/embed-all           # Embed all unembedded documents
GET  /api/documents/                    # List documents
GET  /api/documents/{id}               # Document details
DELETE /api/documents/{id}             # Delete document (cascades chunks/claims)
POST /api/documents/{id}/embed         # Embed single document chunks
POST /api/documents/{id}/extract       # LLM extraction only (concepts/claims/relationships)
POST /api/documents/{id}/process       # Full pipeline: embed + extract
GET  /api/documents/{id}/concepts      # Concepts extracted from a document
GET  /api/documents/{id}/claims        # Claims extracted from a document

GET  /api/search/similar               # Vector similarity search (?query=&top_k=&threshold=)

POST /api/concepts/extract             # Legacy: extract from all docs (background)
POST /api/concepts/normalize           # Deduplicate + merge concepts (synchronous)
POST /api/concepts/build               # Full knowledge rebuild: normalize→cluster→relationships→gaps→consensus
GET  /api/concepts/                    # List concepts (?gaps_only=true&skip=&limit=)
GET  /api/concepts/map                 # React Flow concept map: nodes+edges+gaps+metadata (PRIMARY FRONTEND ENDPOINT)
POST /api/concepts/relationships       # Detect & persist all concept relationships (synchronous)
GET  /api/concepts/relationships       # List relationships (?skip=&limit=, ordered by strength)
POST /api/concepts/detect-gaps         # Run gap detection (synchronous; re-runnable)
GET  /api/concepts/gaps                # List gap concepts (?subtype=expected_topic|bridging|weak_coverage)
GET  /api/concepts/{id}               # Concept detail (aliases, source docs, claims)
POST /api/concepts/cluster             # HDBSCAN clustering + hierarchy (synchronous)

POST /api/brain/build                  # Compute consensus scores + LLM synthesis summary (idempotent)
POST /api/brain/chat                   # RAG chat: embed question → chunk search → concept lookup → LLM answer
GET  /api/brain/status                 # Health snapshot (doc/concept/gap/relationship counts, health_score)
POST /api/brain/rebuild                # Reset consensus scores + clear BrainState + full rebuild

POST /api/pipeline/process-all         # Batch: embed+extract all docs → rebuild knowledge graph
POST /api/pipeline/full-rebuild        # Nuclear: re-embed+re-extract all docs → full knowledge rebuild
```

## ML Pipeline Flow
1. **Parse** document (PyMuPDF/python-docx + OCR fallback)
2. **Chunk** into ~500 token segments with 50 token overlap
3. **Embed** each chunk via Ollama nomic-embed-text (768-dim, L2-normalised, SHA-256 cache)
4. **Extract** concepts + claims + relationships via LLM (qwen2.5:3b), robust 5-stage JSON parser
5. **Normalize** deduplicate concepts: exact-name grouping → AgglomerativeClustering (cosine, threshold 0.85; 0.80 for small projects < 5 concepts) → remap claims/relationships → delete duplicates
6. **Cluster** via HDBSCAN (adaptive `min_cluster_size = max(3, n//10)`); KMeans fallback; noise reassigned to nearest centroid; generality score = `0.4·doc_freq + 0.3·claim_freq + 0.2·specificity + 0.1·centrality`; 2-level hierarchy (highest generality = cluster head)
7. **Detect relationships** — 5-step pipeline: numpy cosine similarity matrix → evidence gathering (claims + co-occurrence) → LLM classification → multi-signal strength scoring → deduplication → persist
8. **Detect gaps** — 3 passes: (1) LLM expected-topic gaps → new Concept rows; (2) graph bridging gaps between disconnected clusters with high centroid similarity → new Concept rows; (3) statistical weak-coverage: existing concepts with coverage < 0.2 and generality > 0.5 flagged in-place
9. **Build brain** — consensus scoring per concept (`support_weight / (support_weight + contradict_weight)`) → LLM synthesis summary → BrainState row
10. **RAG chat** — embed question → pgvector chunk search → pgvector concept search → gather claims/gaps → LLM answer with citations

## Service Details

### `embedding.py` — OllamaEmbeddingService
- `embed_text(text)` / `embed_batch(texts)` — with 3 retries, exponential backoff, SHA-256 cache
- `embed_document_chunks(doc_id, db)` — stores vectors in DB; saves doc-level avg in `metadata_json["document_embedding"]`
- `find_similar_chunks(query_emb, db, top_k, threshold, document_id)` — raw SQL `<=>` pgvector cosine distance
- `asyncio.Semaphore(3)` for concurrency limiting

### `llm_extractor.py` — OllamaLLMService
- `extract_concepts(chunk_text, document_context)` → `List[Dict]` with name, description, generality_score, confidence
- `extract_claims(chunk_text, concepts)` → `List[Dict]` with claim_text, related_concept, claim_type, confidence
- `process_document(doc_id, db)` — full pipeline: per-chunk extraction → exact-name dedup → embedding-similarity dedup → save concepts/claims to DB
- `_parse_json_robust()` — 5-stage parser: direct → strip fences → fix issues → extract structure → add missing bracket
- `asyncio.Semaphore(2)`, 120s timeout, 2 JSON retries per call

### `normalizer.py` — ConceptNormalizer
- `normalize_project(project_id, db)` → `NormalizationResult` dataclass
- Groups by `clean_concept_name()` → AgglomerativeClustering (cosine, threshold 0.85) → picks canonical name (most frequent → shortest → alpha) → remaps claims/relationships/parent refs via bulk UPDATE → deletes duplicates

### `clustering.py` — ConceptClusterer
- `cluster_project(project_id, db)` → `ClusteringResult` dataclass
- HDBSCAN with `metric="euclidean"` on unit vectors (≈ cosine); KMeans fallback if < 2 clusters
- Noise reassigned to nearest cluster centroid; generality score; 2-level hierarchy; bridge detection (O(n²), skipped for n > 500)

### `relationships.py` — RelationshipDetector
- `detect_relationships(project_id, db)` → `RelationshipResult` dataclass
- `MAX_LLM_CALLS=500`, `SIM_LOW=0.30`, `SIM_HIGH=0.85`
- Candidate pairs: numpy `embs @ embs.T`, upper-triangle filter, sorted by similarity descending
- Evidence: claims per concept + co-occurring chunks + shared document count
- Strength: `0.30·sim + 0.25·norm_co_occurrence + 0.25·(shared_docs > 0) + 0.20·llm_confidence`
- LLM → DB type mapping: `supports→SIMILAR`, `extends→BUILDS_ON`, `contradicts→CONTRADICTS`, `complements→COMPLEMENTS`
- Deduplication: canonical key `(min_id, max_id)`; bidirectional `extends+extends → complements`

### `gap_detector.py` — GapDetector
- `detect_gaps(project_id, db)` → `GapDetectionResult` dataclass; re-runnable
- **Pass 1 — Expected topics:** top-25 concepts → LLM prompt → creates synthetic Concept rows (`gap_type=MISSING_LINK`, `metadata_json["is_synthetic_gap"]=True`, `gap_subtype="expected_topic"`)
- **Pass 2 — Bridging gaps:** cluster centroid numpy averages → unconnected cluster pairs with centroid similarity ≥ 0.40 → LLM suggests bridge concept → embedding = normalized average of the two centroids → creates synthetic Concept row (`gap_subtype="bridging"`)
- **Pass 3 — Weak coverage:** flags existing concepts with `coverage_score < 0.2 AND generality_score > 0.5` → `gap_type=UNDER_EXPLORED`, `gap_subtype="weak_coverage"` (skipped if project has < 3 documents)
- Domain inference: joins names of top-8 concepts by generality_score

### `brain_service.py` — BrainService
- `build_brain(project_id, db, *, clear_existing=False)` → `BrainBuildResult` dataclass
  - Consensus formula: `support_weight / (support_weight + contradict_weight)` where supports=1.0, extends=0.7, complements=0.5, contradicts=1.0; neutral 0.5 if no claims
  - Categorises: strong (>0.8), contested (0.3–0.8), contradicted (<0.3)
  - Generates LLM synthesis summary; upserts BrainState row; single commit
- `chat(question, project_id, db, top_k=5)` → `ChatResult` dataclass
  - pgvector chunk search (threshold 0.35, filtered to project docs) + pgvector concept search
  - Gathers claims + top-5 gap nodes → `_CHAT_PROMPT` → LLM → structured result with sources and confidence

### `pipeline.py` — LacunaPipeline
- `process_document(document_id, db)` → `ProcessingResult` dataclass
  - Phase 1: `OllamaEmbeddingService.embed_document_chunks()` — embeds all un-embedded chunks
  - Phase 2: `OllamaLLMService.process_document()` — concept + claim + relationship extraction
  - Returns combined embedding + extraction stats with processing time
- `rebuild_project_knowledge(project_id, db)` → `KnowledgeResult` dataclass
  - [1/5] Normalise via `ConceptNormalizer.normalize_project()`
  - [2/5] Cluster via `ConceptClusterer.cluster_project()`
  - [3/5] Relationships via `RelationshipDetector.detect_relationships()`
  - [4/5] Gaps via `GapDetector.detect_gaps()`
  - [5/5] Brain via `BrainService.build_brain()`
  - Each step wrapped in try/except — failures are logged and zeroed in the result, pipeline continues
- `add_document_and_update(document_id, project_id, db)` → `FullResult` dataclass
  - Convenience wrapper: `process_document` + `rebuild_project_knowledge` in sequence

## Schemas (key additions beyond basics)
- `EmbedDocumentResponse`, `EmbedAllResponse` — embedding phase results
- `SimilarChunkResult`, `SimilarChunksResponse` — vector search results
- `ExtractionResponse`, `ProcessDocumentResponse` — extraction phase results
- `NormalizationResponse` — merge summary (before/after counts, alias_map, groups_merged)
- `ConceptDetailResponse` — concept with aliases, source_documents, claims list
- `ClusterMemberNode` — self-referential node; requires `model_rebuild()`
- `ClusterGroup`, `ConceptMapResponse` — legacy hierarchical map (kept for backward compat)
- `ClusteringResponse` — clustering operation result
- `RelationshipDetectionResponse` — relationship detection result with by_type_count
- `GapItem` — single gap with gap_subtype, importance, suggestions, is_synthetic
- `GapDetectionResponse` — gap detection summary counts
- `BrainBuildResponse` — concepts_scored, consensus breakdown counts, summary_text
- `BrainChatRequest` / `BrainChatResponse` — question + answer with sources, relevant_concepts, confidence
- `BrainStatusResponse` — doc/concept/gap/relationship counts, avg_consensus, health_score, has_brain
- `ReactFlowPosition`, `ReactFlowNodeData`, `ReactFlowNode` — React Flow node shape
- `ReactFlowEdgeData`, `ReactFlowEdge` — React Flow edge shape
- `ReactFlowMapMetadata`, `ReactFlowConceptMapResponse` — primary map response (nodes + edges + gaps + metadata)
- `KnowledgeBuildResponse` — full rebuild summary: all 5 pipeline stages in one response
- `PipelineProcessAllResponse` — batch process summary with documents_failed and per-doc errors list
- `ProcessingResultResponse` — single-document pipeline result shape

## Health Score Formula (`GET /api/brain/status`)
```
health_score = 0.40 * avg_consensus + 0.30 * min(doc_count / 10, 1.0) + 0.30 * min(concept_count / 50, 1.0)
```
- `avg_consensus` defaults to 0.5 (neutral) when no concepts have been scored yet

## Environment Config (.env)
- `DATABASE_URL`: postgresql+asyncpg://...
- `OLLAMA_BASE_URL`: http://localhost:11434
- `OLLAMA_EMBED_MODEL`: nomic-embed-text
- `OLLAMA_LLM_MODEL`: qwen2.5:3b
- `VECTOR_DIMENSION`: 768
- `CHUNK_SIZE`: 500, `CHUNK_OVERLAP`: 50
- `HDBSCAN_MIN_CLUSTER_SIZE`: 3, `HDBSCAN_MIN_SAMPLES`: 2, `HDBSCAN_METRIC`: euclidean
- `CONCEPT_SIMILARITY_THRESHOLD`: 0.85 (normalizer merge threshold)
- `CONCEPT_SIMILARITY_THRESHOLD_SMALL`: 0.80 (for projects with < 5 concepts)
- `CONCEPT_SMALL_PROJECT_THRESHOLD`: 5
- `CLUSTER_BRIDGE_THRESHOLD`: 0.60 (cross-cluster bridge detection)
- `GAP_SIMILARITY_THRESHOLD`: 0.7
- `MIN_COVERAGE_THRESHOLD`: 0.3
- `ALLOWED_ORIGINS`: http://localhost:3000

## Known Bugs Fixed
- **`AssertionError: Status code 204 must not have a response body`** — caused by `from __future__ import annotations` making `-> None` a string. Fix: add `response_model=None` explicitly to any `status_code=204` route decorator.
- **`ObjectNotExecutableError: Not an executable object`** — SQLAlchemy 2.0 rejects bare strings in `conn.execute()`. Fix: wrap with `text()` (already applied to `database.py` `init_db()`).
- **`No module named 'greenlet'`** — SQLAlchemy async requires `greenlet`. Added to `requirements.txt`.

## React Flow Map Format (`GET /api/concepts/map`)
```json
{
  "nodes": [
    {
      "id": "concept_1",
      "type": "concept",
      "data": {
        "label": "Convolutional Neural Networks",
        "description": "...",
        "coverage_score": 0.85,
        "consensus_score": 0.92,
        "generality_score": 0.67,
        "document_count": 7,
        "is_gap": false,
        "gap_type": null,
        "cluster_id": "cluster_0",
        "parent_id": null,
        "children": ["concept_5", "concept_8"]
      },
      "position": {"x": 0, "y": 0}
    }
  ],
  "edges": [
    {
      "id": "rel_1",
      "source": "concept_1",
      "target": "concept_4",
      "type": "builds_on",
      "data": {"strength": 0.78, "confidence": 0.85, "label": "builds_on"}
    }
  ],
  "gaps": [
    {
      "id": 42, "name": "Attention Mechanisms", "gap_type": "missing_link",
      "gap_subtype": "expected_topic", "importance": "critical",
      "suggestions": ["..."], "related_to": ["..."], "is_synthetic": true
    }
  ],
  "metadata": {
    "total_concepts": 45,
    "total_relationships": 23,
    "total_gaps": 5,
    "num_clusters": 4,
    "brain_last_updated": "2026-02-18T12:00:00Z",
    "has_clustering": true
  }
}
```
- `type` is `"gap"` for gap nodes, `"concept"` for real concepts
- `document_count` = `COUNT(DISTINCT document_id)` via claims join
- `position` is always `{x:0, y:0}` — the frontend handles layout
- `gap_type` and `gap_subtype` match the values stored by `GapDetector`

## Frontend Integration Status (as of 2026-02-18)

The Next.js frontend (`../lacuna/`) is now wired to this backend. All communication goes through `lib/api/` in the frontend.

### Connected endpoints
| Frontend action | Backend endpoint |
|---|---|
| Concept map renders | `GET /api/concepts/map` |
| Chat panel sends message | `POST /api/brain/chat` |
| Upload tab (PDF/DOCX) | `POST /api/documents/upload` → `POST /api/documents/{id}/process` |
| Health / status | `GET /api/health` |

### Typical first-use workflow
1. Start the backend: `uvicorn app.main:app --reload --port 8000`
2. Open the frontend at `http://localhost:3000`
3. Open a room → click **Add Papers** → **Upload File** tab → upload PDFs/DOCX
4. Trigger knowledge build: `POST /api/pipeline/process-all` via Swagger at `http://localhost:8000/docs`
5. Refresh the room — the concept map now shows real nodes from the backend
6. Chat with the AI in the Chat tab

### Data flow notes
- All positions in `/api/concepts/map` are `{x:0, y:0}` — correct, the frontend's `useForceLayout` (Cytoscape fcose) handles the actual positioning
- Backend relationship types (`builds_on`, `prerequisite`, `similar`, `parent_child`) are mapped to frontend types (`extends`, `supports`, `complements`) in `lib/api/transform.ts`
- CORS is configured for `http://localhost:3000` and `http://localhost:3001` via `ALLOWED_ORIGINS` in `.env`
- The frontend gracefully falls back to demo scenario data when the backend is unreachable

## Current Limitations
- Single-user (hardcoded `DEFAULT_PROJECT_ID=1`, no auth)
- Background tasks (legacy `/extract`) have no progress tracking
- In-process embedding cache (SHA-256) is lost on server restart
- Bridge relationships detected during clustering are not stored in the relationships table
- No tests
- Bridge detection skipped for projects with > 500 concepts (O(n²) cost)
- `process-all` and `full-rebuild` are synchronous — for large collections callers should expect long response times

## Development Commands
```bash
# Start PostgreSQL + pgvector
docker-compose up -d

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# API docs
http://localhost:8000/docs

# Run Ollama models
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

