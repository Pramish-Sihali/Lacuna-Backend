# Lacuna Backend - Quick Start Guide

Get up and running with Lacuna backend in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] Docker Desktop installed and running
- [ ] Ollama installed ([download here](https://ollama.ai/))

## Step-by-Step Setup

### 1. Install Python Dependencies (2 min)

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

### 2. Start PostgreSQL Database (1 min)

```bash
# Start PostgreSQL with pgvector
docker-compose up -d

# Verify it's running
docker-compose ps
```

You should see the `lacuna_postgres` container running.

### 3. Setup Ollama Models (2 min)

```bash
# Pull embedding model (nomic-embed-text)
ollama pull nomic-embed-text

# Pull LLM model (qwen2.5:3b) - this may take a few minutes
ollama pull qwen2.5:3b

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### 4. Verify Setup

```bash
# Run verification script
python verify_setup.py
```

This will check:
- ✓ Python version
- ✓ All dependencies installed
- ✓ Database connection
- ✓ pgvector extension
- ✓ Ollama service
- ✓ Required models

### 5. Start the Backend

```bash
# Simple way
python run.py

# Or with uvicorn directly
uvicorn app.main:app --reload
```

The API will start at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs ← Open this in your browser!

## Your First API Calls

### 1. Health Check

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "ok",
  "ollama": "ok",
  "timestamp": "2024-..."
}
```

### 2. Upload a Document

```bash
# Using curl
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@path/to/your/document.pdf"

# Or use the interactive docs at http://localhost:8000/docs
```

### 3. Extract Concepts

```bash
curl -X POST "http://localhost:8000/api/concepts/extract"
```

This runs in the background. Wait 30-60 seconds, then check:

```bash
curl "http://localhost:8000/api/concepts/"
```

### 4. Query Your Documents

```bash
curl -X POST "http://localhost:8000/api/brain/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main topics?",
    "top_k": 5
  }'
```

### 5. Get Concept Map

```bash
curl "http://localhost:8000/api/concepts/map"
```

## Interactive API Documentation

The easiest way to explore the API is through the **built-in Swagger UI**:

👉 **Open http://localhost:8000/docs in your browser**

Here you can:
- See all available endpoints
- Try out API calls directly
- View request/response schemas
- Download OpenAPI specification

## Complete Workflow Example

```bash
# 1. Upload documents
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@paper1.pdf"

curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@paper2.pdf"

# 2. Extract concepts (wait ~1 min)
curl -X POST "http://localhost:8000/api/concepts/extract"

# 3. Cluster concepts (wait ~30 sec)
curl -X POST "http://localhost:8000/api/concepts/cluster"

# 4. Detect gaps (wait ~30 sec)
curl -X POST "http://localhost:8000/api/concepts/detect-gaps"

# 5. View concept map
curl "http://localhost:8000/api/concepts/map" | python -m json.tool

# 6. Query knowledge base
curl -X POST "http://localhost:8000/api/brain/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "top_k": 5}'
```

## Troubleshooting

### "Ollama connection failed"
```bash
# Make sure Ollama is running
# Windows: Check in Task Manager
# Mac: Check in Activity Monitor
# Linux: systemctl status ollama

# Or restart Ollama and verify
curl http://localhost:11434/api/tags
```

### "PostgreSQL connection failed"
```bash
# Check Docker container
docker-compose ps

# View logs
docker-compose logs postgres

# Restart
docker-compose restart postgres
```

### "Module not found" errors
```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Database not initialized
```bash
# The app auto-creates tables on first run
# Or manually run:
python -c "from app.database import init_db; import asyncio; asyncio.run(init_db())"
```

## Next Steps

Once everything is running:

1. **Upload your research papers** via `/api/documents/upload`
2. **Extract concepts** to build your knowledge graph
3. **Run clustering** to organize concepts
4. **Detect gaps** to find research opportunities
5. **Query your corpus** using natural language

## Development Tips

- **Auto-reload**: Use `--reload` flag for development (already in `run.py`)
- **Logs**: Check console output for detailed processing logs
- **Database**: Access via `docker exec -it lacuna_postgres psql -U lacuna_user -d lacuna_db`
- **Reset data**: `docker-compose down -v` (warning: deletes all data!)

## Getting Help

- Check the main [README.md](./README.md) for detailed documentation
- View API docs at http://localhost:8000/docs
- Check logs in the console where you ran `python run.py`

---

Happy researching! 🚀
