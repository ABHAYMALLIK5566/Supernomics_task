# Legal Research Assistant - RAG System

A vertical AI agent for law research assistance built with LangChain, FastAPI, and PostgreSQL with pgvector. This system provides intelligent legal document analysis, citation extraction, and domain-specific legal research capabilities.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Document Ingestion](#document-ingestion)
- [Database Management](#database-management)
- [API Usage](#api-usage)
- [Multi-Hop Reasoning](#multi-hop-reasoning)
- [Docker Management](#docker-management)
- [System Testing](#system-testing)
- [Maintenance Commands](#maintenance-commands)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)

## Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- OpenAI API Key
- Git

## Quick Start

```bash
git clone <repository-url>
cd Supernomics_task
cp env.example .env
nano .env
sudo docker-compose up -d
python ingest_sample_docs.py
curl http://localhost:8001/health
```

## Environment Setup

### Create Environment File
```bash
cp env.example .env
```
Creates environment configuration file from template.

### Configure OpenAI API Key
```bash
nano .env
```
Edit the .env file and replace `your-openai-api-key-here` with your actual OpenAI API key.

### Minimal Environment Configuration
```bash
echo "OPENAI_API_KEY=sk-your-actual-api-key-here" > .env
```
Creates minimal .env file with only required OpenAI API key.

### Verify Environment Setup
```bash
grep OPENAI_API_KEY .env
```
Checks if OpenAI API key is properly configured.

## Document Ingestion

### Ingest Sample Documents
```bash
python ingest_sample_docs.py
```
Ingests all PDF files from the sample_documents directory into the vector database.

### Ingest Single PDF File
```bash
curl -X POST "http://localhost:8001/ingest-pdfs" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/your/document.pdf"
```
Uploads and processes a single PDF file through the API.

### Ingest Multiple PDF Files
```bash
curl -X POST "http://localhost:8001/ingest-pdfs" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "files=@document3.pdf"
```
Uploads and processes multiple PDF files simultaneously.

### Check Ingestion Status
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
async def check():
    await db_manager.initialize()
    async with db_manager.get_connection() as conn:
        result = await conn.fetchrow('SELECT COUNT(*) as count FROM documents')
        print(f'Total documents: {result[\"count\"]}')
        result = await conn.fetchrow('SELECT COUNT(*) as count FROM documents WHERE embedding IS NOT NULL')
        print(f'Documents with embeddings: {result[\"count\"]}')
asyncio.run(check())
"
```
Displays total document count and embedding status.

## Database Management

### View All Documents
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
async def view_docs():
    await db_manager.initialize()
    async with db_manager.get_connection() as conn:
        docs = await conn.fetch('SELECT id, title, source, LEFT(content, 100) as preview FROM documents LIMIT 10')
        for doc in docs:
            print(f'ID: {doc[\"id\"]}, Title: {doc[\"title\"]}, Source: {doc[\"source\"]}')
            print(f'Preview: {doc[\"preview\"]}...\n')
asyncio.run(view_docs())
"
```
Lists first 10 documents with basic information.

### Search Documents by Content
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
async def search_docs():
    await db_manager.initialize()
    async with db_manager.get_connection() as conn:
        docs = await conn.fetch(\"SELECT id, title, source FROM documents WHERE content ILIKE '%article%' LIMIT 5\")
        for doc in docs:
            print(f'ID: {doc[\"id\"]}, Title: {doc[\"title\"]}, Source: {doc[\"source\"]}')
asyncio.run(search_docs())
"
```
Finds documents containing specific text patterns.

### View Document Sources
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
async def view_sources():
    await db_manager.initialize()
    async with db_manager.get_connection() as conn:
        sources = await conn.fetch('SELECT DISTINCT source, COUNT(*) as count FROM documents GROUP BY source')
        for source in sources:
            print(f'{source[\"source\"]}: {source[\"count\"]} documents')
asyncio.run(view_sources())
"
```
Shows document distribution by source.

### Delete Document by ID
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
async def delete_doc():
    await db_manager.initialize()
    async with db_manager.get_connection() as conn:
        await conn.execute('DELETE FROM documents WHERE id = 123')
        print('Document deleted')
asyncio.run(delete_doc())
"
```
Removes a specific document from the database.

### Clear All Documents
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
async def clear_docs():
    await db_manager.initialize()
    async with db_manager.get_connection() as conn:
        await conn.execute('DELETE FROM documents')
        print('All documents cleared')
asyncio.run(clear_docs())
"
```
Removes all documents from the database.

### View Database Statistics
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
async def stats():
    await db_manager.initialize()
    async with db_manager.get_connection() as conn:
        total = await conn.fetchval('SELECT COUNT(*) FROM documents')
        with_embeddings = await conn.fetchval('SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL')
        print(f'Total documents: {total}')
        print(f'With embeddings: {with_embeddings}')
        print(f'Embedding coverage: {with_embeddings/total*100:.1f}%')
asyncio.run(stats())
"
```
Displays comprehensive database statistics.

## API Usage

### Health Check
```bash
curl http://localhost:8001/health
```
Verifies that the API service is running and healthy.

### Basic Query (RAG Only)
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are fundamental rights?", "use_agent": false}'
```
Sends a basic query using RAG without agent orchestration.

### Advanced Query (With Agent)
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the fundamental principles of the UN Charter?", "use_agent": true, "top_k": 5}'
```

## Multi-Hop Reasoning

The system now includes advanced multi-hop reasoning capabilities for handling complex legal queries that require iterative analysis and synthesis.

### Features
- **Automatic Complexity Detection**: Analyzes query structure and routes complex queries to multi-hop processing
- **Query Decomposition**: Breaks down complex queries into focused sub-queries
- **Iterative Reasoning**: Executes multiple reasoning steps with intermediate result tracking
- **Intelligent Synthesis**: Combines results from all steps into comprehensive answers
- **Persistent Storage**: Stores complete reasoning chains for future reference

### Complex Query Example
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the enforcement mechanisms in Article 41 and Article 42 of the UN Charter, and explain how they differ from the collective security provisions in Article 51, including the procedural requirements and limitations for each approach.",
    "enable_multi_hop_reasoning": true,
    "session_id": "user_session_123"
  }'
```

### Force Multi-Hop Reasoning
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Article 41 of the UN Charter?",
    "force_multi_hop": true,
    "session_id": "user_session_123"
  }'
```

### Retrieve Reasoning Chains
```bash
# Get reasoning chains for a session
curl -X POST "http://localhost:8001/reasoning-chains" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_session_123",
    "limit": 10
  }'

# Get specific reasoning chain
curl -X GET "http://localhost:8001/reasoning-chains/{chain_id}"

# Get reasoning statistics
curl -X GET "http://localhost:8001/reasoning-statistics?days=30"
```

### Testing Multi-Hop Reasoning
```bash
# Run the comprehensive test suite
python test_multi_hop_reasoning.py
```

For detailed documentation on multi-hop reasoning, see [MULTI_HOP_REASONING.md](MULTI_HOP_REASONING.md).

### Query with Custom Parameters
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain criminal law principles",
    "use_agent": true,
    "top_k": 8,
    "algorithm": "hybrid",
    "similarity_threshold": 0.3
  }'
```
Sends query with custom retrieval parameters.

### Streaming Query
```bash
curl -X POST "http://localhost:8001/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are human rights?", "use_agent": true}'
```
Sends query with real-time streaming response via Server-Sent Events.

### Text-Only Query
```bash
curl -X POST "http://localhost:8001/query-text" \
  -H "Content-Type: application/json" \
  -d '{"query": "Define constitutional law", "use_agent": true}'
```
Sends query and returns only the response text without metadata.

### Query with JSON Formatting
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is international law?", "use_agent": true}' | python3 -m json.tool
```
Sends query and formats the JSON response for readability.

### Batch Query Testing
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the UN Charter principles?", "use_agent": true}' && \
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain human rights", "use_agent": true}' && \
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is criminal law?", "use_agent": true}'
```
Executes multiple queries sequentially for testing.

## Docker Management

### Start All Services
```bash
sudo docker-compose up -d
```
Starts all services (API, PostgreSQL, Redis) in detached mode.

### Start Services with Logs
```bash
sudo docker-compose up
```
Starts all services with live log output.

### Stop All Services
```bash
sudo docker-compose down
```
Stops and removes all containers.

### Restart Services
```bash
sudo docker-compose restart
```
Restarts all running services.

### View Service Status
```bash
sudo docker-compose ps
```
Shows status of all services.

### View Service Logs
```bash
sudo docker-compose logs -f rag-api
```
Shows live logs from the API service.

### View All Service Logs
```bash
sudo docker-compose logs -f
```
Shows live logs from all services.

### Rebuild and Start Services
```bash
sudo docker-compose up --build -d
```
Rebuilds Docker images and starts services.

### Stop and Remove Everything
```bash
sudo docker-compose down -v
```
Stops services and removes volumes (deletes database data).

### Execute Commands in API Container
```bash
sudo docker-compose exec rag-api bash
```
Opens an interactive bash session in the API container.

### Execute Python Commands in Container
```bash
sudo docker-compose exec rag-api python -c "print('Hello from container')"
```
Runs Python commands inside the API container.

### Copy Files to Container
```bash
sudo docker cp local_file.py task_supernomics-rag-api:/app/
```
Copies files from host to the API container.

### Copy Files from Container
```bash
sudo docker cp task_supernomics-rag-api:/app/logs.txt ./
```
Copies files from container to host.

## System Testing

### Run Complete System Test
```bash
python test_system.py
```
Executes comprehensive system tests including database, RAG, and agent functionality.

### Test Database Connection
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
async def test():
    await db_manager.initialize()
    healthy = await db_manager.health_check()
    print('Database healthy:', healthy)
asyncio.run(test())
"
```
Tests database connectivity and health.

### Test RAG Engine
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.services.lightweight_llm_rag import lightweight_llm_rag
async def test():
    await lightweight_llm_rag.initialize()
    result = await lightweight_llm_rag.query('test query', top_k=3)
    print('RAG test successful:', len(result.get('sources', [])) > 0)
asyncio.run(test())
"
```
Tests the RAG engine functionality.

### Test Legal Classifier
```bash
sudo docker-compose exec rag-api python -c "
from app.services.legal_classifier import legal_classifier
result = legal_classifier.classify('The defendant is charged with murder')
print('Classification:', result)
"
```
Tests the legal text classification model.

### Test LangChain Agent
```bash
sudo docker-compose exec rag-api python -c "
import asyncio
from app.services.langchain_agent import langchain_legal_agent
async def test():
    await langchain_legal_agent.initialize()
    result = await langchain_legal_agent.research('What is law?')
    print('Agent test successful:', result.domain)
asyncio.run(test())
"
```
Tests the LangChain legal research agent.

### Performance Test
```bash
time curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are fundamental rights?", "use_agent": true}'
```
Measures API response time for performance testing.

## Maintenance Commands

### Clear Redis Cache
```bash
sudo docker-compose exec redis redis-cli FLUSHALL
```
Clears all cached data from Redis.

### View Redis Cache Status
```bash
sudo docker-compose exec redis redis-cli INFO memory
```
Shows Redis memory usage and cache statistics.

### Backup Database
```bash
sudo docker-compose exec postgres pg_dump -U postgres ragdb > backup.sql
```
Creates a backup of the PostgreSQL database.

### Restore Database
```bash
sudo docker-compose exec -T postgres psql -U postgres ragdb < backup.sql
```
Restores database from backup file.

### View Database Size
```bash
sudo docker-compose exec postgres psql -U postgres ragdb -c "
SELECT pg_size_pretty(pg_database_size('ragdb')) as database_size;
"
```
Shows the size of the database.

### Optimize Database
```bash
sudo docker-compose exec postgres psql -U postgres ragdb -c "VACUUM ANALYZE;"
```
Optimizes database performance and updates statistics.

### View Container Resource Usage
```bash
sudo docker stats
```
Shows resource usage (CPU, memory) for all containers.

### Clean Up Docker Resources
```bash
sudo docker system prune -f
```
Removes unused Docker resources to free up space.

### Update Dependencies
```bash
sudo docker-compose exec rag-api pip install --upgrade -r requirements.txt
```
Updates Python dependencies in the container.

### View API Documentation
```bash
curl http://localhost:8001/docs
```
Accesses the interactive API documentation (Swagger UI).

### Check Service Health
```bash
curl http://localhost:8001/health | python3 -m json.tool
```
Checks detailed health status of all services.

## Architecture

The system consists of:

- **FastAPI Application**: REST API with streaming support
- **LangChain Agent**: Multi-step legal reasoning with tools
- **PostgreSQL + pgvector**: Vector database for document storage
- **Redis**: Caching layer for performance
- **OpenAI Integration**: LLM and embedding services
- **Legal Tools**: Citation extraction and domain classification

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /query` - Main query endpoint
- `POST /query-text` - Text-only response
- `POST /stream` - Streaming responses
- `POST /ingest-pdfs` - Document ingestion

### Response Format
```json
{
  "response": "Legal analysis text...",
  "query": "User query",
  "context": [...],
  "metadata": {
    "algorithm": "langchain_agent",
    "citations": ["Article 1", "Section 2"],
    "domain": "Constitutional Law",
    "confidence": 0.85,
    "tools_used": ["legal_research", "extract_legal_citations", "classify_legal_domain"]
  },
  "source": "legal_agent",
  "response_time_ms": 1250
}
```

### Streaming Format
```
data: {"status": "processing", "message": "Starting legal research..."}
data: {"delta": "Response chunk", "type": "content"}
data: {"type": "metadata", "citations": [...], "domain": "..."}
data: {"type": "done", "status": "completed"}
```
