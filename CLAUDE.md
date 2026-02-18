# Lacuna Backend - Project Context

## What is Lacuna?
AI-powered research discovery platform that helps academics visualize knowledge gaps in document collections. Users upload research papers (PDF/DOCX), the system extracts concepts, detects relationships, identifies knowledge gaps, and provides RAG querying.

## Tech Stack
- **Framework:** FastAPI 0.115 (async, Python 3.11+)
- **Server:** Uvicorn (ASGI)
- **Database:** PostgreSQL 16 + pgvector extension (vector similarity search)
- **ORM:** SQLAlchemy 2.0 (async) with Alembic migrations
- **LLM:** Ollama local inference (qwen2.5:3b for generation, nomic-embed-text for 768-dim embeddings)
- **Clustering:** HDBSCAN (density-based, no K required)
- **Doc Parsing:** PyMuPDF (PDF), python-docx (DOCX), Tesseract OCR (fallback)
- **Validation:** Pydantic v2 for request/response schemas
- **Frontend:** Next.js (separate repo)

## Project Structure
```
app/
├── main.py                 # FastAPI app entry, CORS, lifespan events, routers
├── config.py               # Pydantic Settings from .env
├── database.py             # Async engine, session factory, init_db()
├── models/
│   ├── database_models.py  # 7 SQLAlchemy tables: projects, documents, chunks, concepts, claims, relationships, brain_state
│   └── schemas.py          # Pydantic request/response models
├── routers/
│   ├── health.py           # GET /api/health (DB + Ollama status)
│   ├── documents.py        # POST upload, GET list, GET by id, DELETE
│   ├── concepts.py         # POST extract, GET list, GET /map, POST cluster, POST detect-gaps
│   └── brain.py            # POST query (RAG), GET consensus, POST build-consensus
├── services/
│   ├── document_parser.py  # PDF/DOCX text extraction with OCR fallback
│   ├── chunking.py         # Semantic chunking (500 tokens, 50 overlap)
│   ├── embedding.py        # Ollama embedding generation, batch processing, similarity search
│   ├── llm_extractor.py    # LLM-based concept/claim/relationship extraction (JSON output)
│   ├── normalizer.py       # Concept dedup (exact name + embedding similarity >0.85)
│   ├── clustering.py       # HDBSCAN clustering + hierarchy detection via generality scores
│   ├── relationships.py    # Relationship detection (6 types) + importance scoring
│   ├── gap_detector.py     # 4 gap types: missing_link, under_explored, contradictory, isolated
│   └── brain_service.py    # RAG pipeline (embed query -> search chunks/concepts -> LLM answer) + consensus
└── utils/
    └── helpers.py          # cosine_similarity, text normalization, keyword extraction, hashing
```

## Database Schema (7 tables)
- **projects** - Research project containers (id, name, description)
- **documents** - Uploaded files + parsed text (FK: project_id)
- **chunks** - Text segments with 768-dim pgvector embeddings (FK: document_id)
- **concepts** - Extracted concepts with scores (generality, coverage, consensus), gap flags, cluster labels, parent-child hierarchy (FK: project_id)
- **claims** - Evidence linking documents to concepts with claim_type (supports|contradicts|extends|complements) (FK: document_id, concept_id)
- **relationships** - Concept-to-concept connections with type (prerequisite|builds_on|contradicts|complements|similar|parent_child), strength, confidence (FK: source/target concept)
- **brain_state** - Project-wide consensus tracking (FK: project_id)

## Key Architecture Patterns
- **Async-first:** All DB and I/O operations use async/await
- **Dependency Injection:** `Depends(get_db)` provides DB sessions to endpoints
- **Background Tasks:** Long operations (extract, cluster, detect-gaps) use FastAPI BackgroundTasks
- **Service Layer:** Routers handle HTTP, services handle business logic
- **Cascade Deletes:** Deleting a document cascades to chunks and claims
- **Batch Embedding:** Embeddings generated in batches of 10

## API Endpoints
```
GET  /api/health                  # System status
POST /api/documents/upload        # Upload + parse + chunk + embed
GET  /api/documents/              # List documents
GET  /api/documents/{id}          # Document details
DELETE /api/documents/{id}        # Delete document
POST /api/concepts/extract        # LLM concept extraction (background)
GET  /api/concepts/               # List concepts (?gaps_only=true)
GET  /api/concepts/map            # Full concept map (nodes + edges)
POST /api/concepts/cluster        # HDBSCAN clustering (background)
POST /api/concepts/detect-gaps    # Gap detection (background)
POST /api/brain/query             # RAG query
GET  /api/brain/consensus         # Current consensus
POST /api/brain/build-consensus   # Build consensus (background)
```

## ML Pipeline Flow
1. **Parse** document (PyMuPDF/python-docx + OCR fallback)
2. **Chunk** into ~500 token segments with 50 token overlap
3. **Embed** each chunk via Ollama nomic-embed-text (768-dim)
4. **Extract** concepts + claims via LLM (qwen2.5:3b)
5. **Normalize** deduplicate concepts (name matching + embedding similarity >0.85)
6. **Cluster** via HDBSCAN (min_cluster_size=3, min_samples=2)
7. **Detect relationships** (embedding similarity + heuristics + optional LLM)
8. **Detect gaps** (missing links, under-explored, contradictory, isolated)
9. **RAG** queries: embed query -> vector search chunks+concepts -> LLM answer with context

## Environment Config (.env)
- DATABASE_URL: postgresql+asyncpg://...
- OLLAMA_BASE_URL: http://localhost:11434
- OLLAMA_EMBED_MODEL: nomic-embed-text
- OLLAMA_LLM_MODEL: qwen2.5:3b
- VECTOR_DIMENSION: 768
- CHUNK_SIZE: 500, CHUNK_OVERLAP: 50
- HDBSCAN_MIN_CLUSTER_SIZE: 3
- ALLOWED_ORIGINS: http://localhost:3000

## Current Limitations
- Single-user (hardcoded DEFAULT_PROJECT_ID=1, no auth)
- Background tasks have no progress tracking
- No caching for embedding queries
- No tests
- No retry logic for Ollama failures

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
