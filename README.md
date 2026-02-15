# Lacuna Backend

Backend API for **Lacuna** — a research discovery platform that helps academics visualize knowledge gaps in their document collections.

## Features

- **Document Processing**: PDF and DOCX parsing with OCR support
- **AI-Powered Analysis**: Concept and claim extraction using Ollama LLM
- **Vector Embeddings**: Semantic search with nomic-embed-text via Ollama
- **Knowledge Graph**: Concept relationships and hierarchy detection
- **Clustering**: HDBSCAN-based concept clustering
- **Gap Detection**: Automatic identification of knowledge gaps
- **RAG System**: Query your documents with context-aware answers
- **Consensus Building**: Cross-document agreement analysis

## Tech Stack

- **FastAPI** - Modern async web framework
- **PostgreSQL + pgvector** - Vector database for embeddings
- **Ollama** - Local LLM and embeddings (qwen2.5:3b, nomic-embed-text)
- **SQLAlchemy** - Async ORM
- **PyMuPDF (fitz)** - PDF parsing
- **Tesseract** - OCR support
- **HDBSCAN** - Density-based clustering

## Prerequisites

- Python 3.11 or higher
- PostgreSQL 16+
- Docker and Docker Compose (for PostgreSQL + pgvector)
- [Ollama](https://ollama.ai/) running locally

## Setup Instructions

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start PostgreSQL with pgvector

```bash
# Start PostgreSQL container
docker-compose up -d

# Verify PostgreSQL is running
docker-compose ps
```

### 3. Setup Ollama

```bash
# Install Ollama from https://ollama.ai/

# Pull required models
ollama pull nomic-embed-text
ollama pull qwen2.5:3b

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### 4. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (defaults should work for local development)
```

### 5. Initialize Database

```bash
# Create database tables
# The app will automatically create tables on startup
# Or use Alembic for migrations:
alembic upgrade head
```

### 6. Run the Application

```bash
# Development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python -m app.main
```

The API will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /api/health` - Check system status

### Documents
- `POST /api/documents/upload` - Upload and process a document
- `GET /api/documents/` - List all documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete a document

### Concepts
- `POST /api/concepts/extract` - Extract concepts from documents (background)
- `GET /api/concepts/` - List all concepts
- `GET /api/concepts/map` - Get concept map with relationships
- `POST /api/concepts/cluster` - Cluster concepts (background)
- `POST /api/concepts/detect-gaps` - Detect knowledge gaps (background)

### Brain (RAG)
- `POST /api/brain/query` - Query the knowledge base
- `GET /api/brain/consensus` - Get consensus state
- `POST /api/brain/build-consensus` - Build consensus from documents

## Usage Workflow

### 1. Upload Documents

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@research_paper.pdf"
```

### 2. Extract Concepts

```bash
curl -X POST "http://localhost:8000/api/concepts/extract"
```

### 3. Cluster Concepts

```bash
curl -X POST "http://localhost:8000/api/concepts/cluster"
```

### 4. Detect Gaps

```bash
curl -X POST "http://localhost:8000/api/concepts/detect-gaps"
```

### 5. Query Knowledge Base

```bash
curl -X POST "http://localhost:8000/api/brain/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main concepts in machine learning?",
    "top_k": 5,
    "include_gaps": true
  }'
```

### 6. Get Concept Map

```bash
curl "http://localhost:8000/api/concepts/map"
```

## Project Structure

```
lacuna-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration settings
│   ├── database.py          # Database connection
│   ├── models/
│   │   ├── database_models.py  # SQLAlchemy models
│   │   └── schemas.py          # Pydantic schemas
│   ├── routers/
│   │   ├── documents.py     # Document endpoints
│   │   ├── concepts.py      # Concept endpoints
│   │   ├── brain.py         # Brain/RAG endpoints
│   │   └── health.py        # Health check
│   ├── services/
│   │   ├── document_parser.py  # PDF/DOCX parsing
│   │   ├── chunking.py         # Text chunking
│   │   ├── embedding.py        # Ollama embeddings
│   │   ├── llm_extractor.py    # Concept extraction
│   │   ├── normalizer.py       # Deduplication
│   │   ├── clustering.py       # HDBSCAN clustering
│   │   ├── relationships.py    # Relationship detection
│   │   ├── gap_detector.py     # Gap detection
│   │   └── brain_service.py    # RAG service
│   └── utils/
│       └── helpers.py       # Utility functions
├── uploads/                 # Document storage
├── alembic/                 # Database migrations
├── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

## Configuration

Key settings in `.env`:

```env
# Database
DATABASE_URL=postgresql+asyncpg://lacuna_user:lacuna_password@localhost:5432/lacuna_db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=qwen2.5:3b

# Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_FILE_SIZE=52428800  # 50MB

# Clustering
HDBSCAN_MIN_CLUSTER_SIZE=3
```

## Database Schema

### Core Tables

- **projects** - Research projects
- **documents** - Uploaded documents with parsed text
- **chunks** - Text chunks with embeddings (768-dim vectors)
- **concepts** - Extracted concepts with scores and embeddings
- **claims** - Document claims supporting concepts
- **relationships** - Concept-to-concept relationships
- **brain_state** - Project-wide consensus tracking

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Code Quality

```bash
# Format code
black app/

# Lint
ruff check app/
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
# Windows: Check Task Manager
# Linux/Mac: systemctl restart ollama
```

### PostgreSQL Issues

```bash
# Check container logs
docker-compose logs postgres

# Restart container
docker-compose restart postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

### Out of Memory

- Reduce `CHUNK_SIZE` in `.env`
- Process documents in smaller batches
- Increase Docker memory allocation

## Performance Optimization

- **Batch Processing**: Process embeddings in batches of 10-20
- **Caching**: Consider adding Redis for frequently accessed data
- **Database Indexing**: Already optimized with indexes on key fields
- **Async Operations**: All I/O operations are async
- **Background Tasks**: Long-running operations use FastAPI background tasks

## Security Notes

- Currently single-user system (hardcoded DEFAULT_PROJECT_ID=1)
- Add authentication before production deployment
- Validate file uploads for malicious content
- Use environment variables for sensitive configuration
- Enable HTTPS in production

## Roadmap

- [ ] Multi-user support with authentication
- [ ] Real-time processing updates via WebSockets
- [ ] Export concept maps to various formats
- [ ] Advanced graph visualization data
- [ ] Citation extraction and tracking
- [ ] Integration with reference managers
- [ ] Cloud deployment configurations

## License

MIT License - see LICENSE file for details

## Support

For issues and questions, please open an issue on GitHub or contact the development team.

---

Built with ❤️ for researchers by researchers
