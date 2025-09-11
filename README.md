# Legal Research Assistant - Advanced RAG System

A sophisticated vertical AI agent for legal research assistance built with LangChain, FastAPI, PostgreSQL with pgvector, and advanced multi-hop reasoning capabilities. This system provides intelligent legal document analysis, citation extraction, domain-specific classification, and comprehensive legal research capabilities.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Document Ingestion](#document-ingestion)
- [API Usage](#api-usage)
- [Multi-Hop Reasoning](#multi-hop-reasoning)
- [Database Management](#database-management)
- [Docker Management](#docker-management)
- [System Testing](#system-testing)
- [Maintenance Commands](#maintenance-commands)
- [Codebase Structure](#codebase-structure)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Performance & Scaling](#performance--scaling)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This Legal Research Assistant is a production-ready RAG (Retrieval-Augmented Generation) system specifically designed for legal research tasks. It combines multiple AI technologies to provide accurate, well-sourced legal information with advanced reasoning capabilities.

### Core Capabilities
- **Intelligent Document Processing**: PDF ingestion with text extraction and vectorization
- **Advanced RAG Engine**: Hybrid search combining vector similarity and keyword matching
- **Multi-Hop Reasoning**: Complex query decomposition and iterative reasoning
- **Legal Domain Classification**: Automatic categorization of legal text
- **Citation Extraction**: Automatic identification and extraction of legal citations
- **Real-time Streaming**: Server-Sent Events for live response streaming
- **Comprehensive Caching**: Redis-based caching for optimal performance
- **Production Ready**: Docker containerization with health checks and monitoring

## Key Features

### AI-Powered Research
- **LangChain Agent Integration**: Multi-step legal reasoning with specialized tools
- **OpenAI GPT-4 Integration**: Advanced language understanding and generation
- **Vector Search**: Semantic similarity search using OpenAI embeddings
- **Hybrid Retrieval**: Combines vector search with keyword matching

### Multi-Hop Reasoning
- **Automatic Complexity Detection**: Analyzes query structure and routes appropriately
- **Query Decomposition**: Breaks complex queries into focused sub-queries
- **Iterative Processing**: Multi-step reasoning with intermediate result tracking
- **Intelligent Synthesis**: Combines results from all reasoning steps
- **Persistent Storage**: Stores complete reasoning chains for future reference

### Legal-Specific Features
- **Domain Classification**: Categorizes text into Constitutional, Criminal, Contract, or Other law
- **Citation Extraction**: Identifies and extracts legal citations and references
- **Legal Tools**: Specialized tools for legal research and analysis
- **Source Attribution**: Comprehensive source tracking and citation

### Performance & Scalability
- **Redis Caching**: Multi-layer caching for optimal response times
- **Async Processing**: Non-blocking operations for high concurrency
- **Connection Pooling**: Efficient database connection management
- **Rate Limiting**: Built-in protection against abuse
- **Health Monitoring**: Comprehensive health checks and metrics

## Architecture

![Architecture Diagram](mermaid/Architecture.png)

### System Components

#### Core Services
- **FastAPI Application**: REST API with streaming support and comprehensive error handling
- **LangChain Agent**: Multi-step legal reasoning with specialized tools
- **Multi-Hop Reasoning Engine**: Complex query processing with iterative analysis
- **Lightweight RAG Engine**: Efficient retrieval and generation system

#### Data Layer
- **PostgreSQL + pgvector**: Vector database for document storage and similarity search
- **Redis**: High-performance caching layer for queries and responses
- **Document Storage**: Structured storage for legal documents and metadata

#### AI/ML Components
- **OpenAI Integration**: GPT-4 for text generation and text-embedding-3-small for embeddings
- **Legal Classifier**: Scikit-learn based domain classification
- **Legal Tools**: Specialized tools for citation extraction and legal analysis

## Prerequisites

- **Docker & Docker Compose**: For containerized deployment
- **Python 3.10+**: For local development
- **OpenAI API Key**: Required for AI functionality
- **Git**: For version control
- **8GB+ RAM**: Recommended for optimal performance
- **10GB+ Storage**: For document storage and vector embeddings

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/ABHAYMALLIK5566/Supernomics_task
cd Supernomics_task
cp env.example .env
```

### 2. Configure Environment
```bash
nano .env
# Add your OpenAI API key:
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Start Services
```bash
sudo docker-compose up -d
```

### 4. Ingest Sample Documents
```bash
python ingest_sample_docs.py
```

### 5. Test the System
```bash
curl http://localhost:8001/health
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are fundamental rights?", "use_agent": true}'
```

## Environment Setup

### Environment File Configuration
```bash
# Copy template
cp env.example .env

# Edit configuration
nano .env
```

### Required Environment Variables
```bash
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Database Configuration
DATABASE_URL=postgresql://postgres:ragpassword@postgres:5432/ragdb
REDIS_URL=redis://redis:6379

# Application Settings
DEBUG=true
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
```

### Verify Configuration
```bash
# Check OpenAI API key
grep OPENAI_API_KEY .env

# Test database connection
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

## Document Ingestion

### Ingest Sample Documents
```bash
python ingest_sample_docs.py
```
Processes all PDF files from the `sample_documents/` directory.

### Ingest Single PDF via API
```bash
curl -X POST "http://localhost:8001/ingest-pdfs" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/your/document.pdf"
```

### Ingest Multiple PDFs
```bash
curl -X POST "http://localhost:8001/ingest-pdfs" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "files=@document3.pdf"
```

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

## API Usage

### Basic Query (RAG Only)
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are fundamental rights?", "use_agent": false}'
```

### Advanced Query (With Agent)
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the fundamental principles of the UN Charter?", 
    "use_agent": true, 
    "top_k": 5
  }'
```

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

### Streaming Query
```bash
curl -X POST "http://localhost:8001/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are human rights?", "use_agent": true}'
```

### Text-Only Response
```bash
curl -X POST "http://localhost:8001/query-text" \
  -H "Content-Type: application/json" \
  -d '{"query": "Define constitutional law", "use_agent": true}'
```

## Multi-Hop Reasoning

The system includes advanced multi-hop reasoning capabilities for handling complex legal queries that require iterative analysis and synthesis.

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

### Database Statistics
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

## Docker Management

### Start All Services
```bash
sudo docker-compose up -d
```

### Start Services with Logs
```bash
sudo docker-compose up
```

### Stop All Services
```bash
sudo docker-compose down
```

### Restart Services
```bash
sudo docker-compose restart
```

### View Service Status
```bash
sudo docker-compose ps
```

### View Service Logs
```bash
# API service logs
sudo docker-compose logs -f rag-api

# All services logs
sudo docker-compose logs -f

# Database logs
sudo docker-compose logs -f postgres

# Redis logs
sudo docker-compose logs -f redis
```

### Rebuild and Start Services
```bash
sudo docker-compose up --build -d
```

### Stop and Remove Everything
```bash
sudo docker-compose down -v
```

### Execute Commands in Containers
```bash
# API container
sudo docker-compose exec rag-api bash

# Database container
sudo docker-compose exec postgres psql -U postgres ragdb

# Redis container
sudo docker-compose exec redis redis-cli
```

## System Testing

### Run Complete System Test
```bash
python test_system.py
```

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

### Test Legal Classifier
```bash
sudo docker-compose exec rag-api python -c "
from app.services.legal_classifier import legal_classifier
result = legal_classifier.classify('The defendant is charged with murder')
print('Classification:', result)
"
```

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

### Performance Test
```bash
time curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are fundamental rights?", "use_agent": true}'
```

## Maintenance Commands

### Clear Redis Cache
```bash
sudo docker-compose exec redis redis-cli FLUSHALL
```

### View Redis Cache Status
```bash
sudo docker-compose exec redis redis-cli INFO memory
```

### Backup Database
```bash
sudo docker-compose exec postgres pg_dump -U postgres ragdb > backup.sql
```

### Restore Database
```bash
sudo docker-compose exec -T postgres psql -U postgres ragdb < backup.sql
```

### View Database Size
```bash
sudo docker-compose exec postgres psql -U postgres ragdb -c "
SELECT pg_size_pretty(pg_database_size('ragdb')) as database_size;
"
```

### Optimize Database
```bash
sudo docker-compose exec postgres psql -U postgres ragdb -c "VACUUM ANALYZE;"
```

### View Container Resource Usage
```bash
sudo docker stats
```

### Clean Up Docker Resources
```bash
sudo docker system prune -f
```

### Update Dependencies
```bash
sudo docker-compose exec rag-api pip install --upgrade -r requirements.txt
```

### View API Documentation
```bash
curl http://localhost:8001/docs
```

### Check Service Health
```bash
curl http://localhost:8001/health | python3 -m json.tool
```

## Codebase Structure

```
Supernomics_task/
├── app/                          # Main application package
│   ├── api/                      # API layer
│   │   ├── __init__.py
│   │   └── endpoints.py          # FastAPI route definitions
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management
│   │   ├── database.py           # Database connection and management
│   │   ├── exceptions.py         # Custom exception classes
│   │   ├── lifecycle.py          # Application lifecycle management
│   │   ├── rate_limiter.py       # Rate limiting implementation
│   │   ├── response_formatter.py # Response formatting utilities
│   │   └── utils.py              # General utilities
│   ├── models/                   # Pydantic models
│   │   ├── __init__.py
│   │   ├── document.py           # Document data models
│   │   └── requests.py           # Request/response models
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── cache.py              # Redis caching service
│   │   ├── langchain_agent.py    # LangChain agent implementation
│   │   ├── legal_classifier.py   # Legal text classification
│   │   ├── legal_tools.py        # Legal-specific tools
│   │   ├── lightweight_llm_rag.py # Core RAG engine
│   │   ├── multi_hop_reasoning.py # Multi-hop reasoning system
│   │   ├── pdf_ingestion.py      # PDF processing service
│   │   ├── query_complexity_detector.py # Query complexity analysis
│   │   └── reasoning_chain_storage.py # Reasoning chain storage
│   └── main.py                   # FastAPI application entry point
├── docker/                       # Docker configuration
│   ├── Dockerfile               # API container definition
│   ├── postgres/
│   │   ├── init.sql             # Database initialization
│   │   └── postgresql.conf/     # PostgreSQL configuration
│   └── redis/
│       └── redis.conf           # Redis configuration
├── sample_documents/            # Sample legal documents
│   ├── coretreatiesen.pdf
│   ├── crc.pdf
│   ├── eng.pdf
│   └── uncharter.pdf
├── docker-compose.yml           # Docker Compose configuration
├── env.example                  # Environment variables template
├── requirements.txt             # Python dependencies
├── test_multi_hop_reasoning.py  # Multi-hop reasoning tests
├── MULTI_HOP_REASONING.md       # Multi-hop reasoning documentation
└── README.md                    # This file
```

### Key Components

#### API Layer (`app/api/`)
- **endpoints.py**: FastAPI route definitions with comprehensive error handling
- Supports query processing, document ingestion, health checks, and streaming
- Includes rate limiting and request validation

#### Core Services (`app/core/`)
- **config.py**: Centralized configuration with environment variable support
- **database.py**: PostgreSQL connection management with pgvector support
- **exceptions.py**: Custom exception classes for error handling
- **lifecycle.py**: Application startup and shutdown management
- **rate_limiter.py**: Redis-based rate limiting implementation
- **response_formatter.py**: Standardized response formatting
- **utils.py**: General utility functions and helpers

#### Models (`app/models/`)
- **document.py**: Document data models and database schemas
- **requests.py**: Pydantic models for request/response validation

#### Services (`app/services/`)
- **lightweight_llm_rag.py**: Core RAG engine with OpenAI integration
- **multi_hop_reasoning.py**: Advanced multi-hop reasoning system
- **langchain_agent.py**: LangChain agent with legal tools
- **legal_classifier.py**: ML-based legal text classification
- **legal_tools.py**: Legal-specific analysis tools
- **pdf_ingestion.py**: PDF processing and text extraction
- **query_complexity_detector.py**: Query complexity analysis
- **reasoning_chain_storage.py**: Reasoning chain persistence
- **cache.py**: Redis caching service

## API Endpoints

### Core Endpoints

#### Health & Information
- `GET /` - Service information and available endpoints
- `GET /health` - Comprehensive health check
- `GET /docs` - Interactive API documentation (Swagger UI)

#### Query Processing
- `POST /query` - Main query endpoint with full response
- `POST /query-text` - Text-only response endpoint
- `POST /stream` - Server-Sent Events streaming endpoint

#### Document Management
- `POST /ingest-pdfs` - PDF document ingestion
- `GET /documents` - List ingested documents
- `DELETE /documents/{doc_id}` - Delete specific document

#### Multi-Hop Reasoning
- `POST /reasoning-chains` - Get reasoning chains for session
- `GET /reasoning-chains/{chain_id}` - Get specific reasoning chain
- `GET /reasoning-statistics` - Get reasoning statistics

### Request/Response Formats

#### Query Request
```json
{
  "query": "What are fundamental rights?",
  "use_agent": true,
  "top_k": 5,
  "algorithm": "hybrid",
  "similarity_threshold": 0.3,
  "enable_multi_hop_reasoning": false,
  "force_multi_hop": false,
  "session_id": "optional_session_id"
}
```

#### Query Response
```json
{
  "response": "Legal analysis text...",
  "query": "User query",
  "context": [
    {
      "content": "Relevant document content...",
      "source": "document.pdf",
      "page": 1,
      "similarity": 0.85
    }
  ],
  "metadata": {
    "algorithm": "langchain_agent",
    "citations": ["Article 1", "Section 2"],
    "domain": "Constitutional Law",
    "confidence": 0.85,
    "tools_used": ["legal_research", "extract_legal_citations"],
    "execution_time_ms": 1250,
    "complexity_level": "moderate"
  },
  "source": "legal_agent",
  "response_time_ms": 1250
}
```

#### Streaming Response Format
```
data: {"status": "processing", "message": "Starting legal research..."}
data: {"delta": "Response chunk", "type": "content"}
data: {"type": "metadata", "citations": [...], "domain": "..."}
data: {"type": "done", "status": "completed"}
```

## Configuration

### Environment Variables

#### Required
- `OPENAI_API_KEY`: OpenAI API key (required)

#### Database
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

#### OpenAI
- `OPENAI_MODEL`: Model for text generation (default: gpt-4-turbo-preview)
- `OPENAI_EMBEDDING_MODEL`: Model for embeddings (default: text-embedding-3-small)

#### Application
- `DEBUG`: Enable debug mode (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

#### RAG Configuration
- `RAG_TOP_K`: Number of documents to retrieve (default: 5)
- `RAG_SIMILARITY_THRESHOLD`: Similarity threshold (default: 0.7)
- `RAG_MAX_TOKENS`: Maximum tokens for generation (default: 4000)

#### Caching
- `CACHE_TTL_SECONDS`: Cache time-to-live (default: 300)
- `CACHE_MAX_QUERY_LENGTH`: Maximum query length for caching (default: 1000)

#### Rate Limiting
- `RATE_LIMIT_REQUESTS`: Requests per window (default: 10)
- `RATE_LIMIT_WINDOW`: Time window (default: 1/minute)

### Configuration File
The system uses Pydantic Settings for configuration management with automatic validation and type conversion. All settings can be overridden via environment variables.

## Performance & Scaling

### Performance Metrics
- **Simple Queries**: < 2 seconds response time
- **Complex Queries**: 5-15 seconds response time
- **Multi-Hop Reasoning**: 15-30 seconds response time
- **Cached Queries**: < 100ms response time

### Optimization Features
- **Multi-layer Caching**: Query results, embeddings, and responses
- **Connection Pooling**: Efficient database connections
- **Async Processing**: Non-blocking operations
- **Parallel Processing**: Concurrent sub-query execution
- **Vector Indexing**: Optimized similarity search

### Scaling Considerations
- **Horizontal Scaling**: Load balancer with multiple API instances
- **Database Scaling**: Read replicas and connection pooling
- **Cache Scaling**: Redis cluster for high availability
- **Storage Scaling**: Distributed vector storage

### Monitoring
- **Health Checks**: Comprehensive service monitoring
- **Metrics**: Performance and usage statistics
- **Logging**: Structured logging with different levels
- **Alerting**: Error and performance threshold alerts

## Troubleshooting

### Common Issues

#### 1. OpenAI API Errors
```bash
# Check API key
grep OPENAI_API_KEY .env

# Test API connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

#### 2. Database Connection Issues
```bash
# Check database status
sudo docker-compose exec postgres pg_isready -U postgres

# Test connection
sudo docker-compose exec rag-api python -c "
import asyncio
from app.core.database import db_manager
asyncio.run(db_manager.health_check())
"
```

#### 3. Redis Connection Issues
```bash
# Check Redis status
sudo docker-compose exec redis redis-cli ping

# Check Redis memory
sudo docker-compose exec redis redis-cli INFO memory
```

#### 4. Slow Response Times
- Check database performance
- Verify OpenAI API response times
- Monitor system resources
- Review cache hit rates

#### 5. Memory Issues
```bash
# Check container memory usage
sudo docker stats

# Check database size
sudo docker-compose exec postgres psql -U postgres ragdb -c "
SELECT pg_size_pretty(pg_database_size('ragdb'));
"
```

### Debug Mode
Enable debug logging for detailed troubleshooting:

```bash
# Set debug environment variable
export LOG_LEVEL=DEBUG

# Restart services
sudo docker-compose restart rag-api
```

### Log Analysis
```bash
# View API logs
sudo docker-compose logs -f rag-api

# View database logs
sudo docker-compose logs -f postgres

# View Redis logs
sudo docker-compose logs -f redis
```
That's all Thank You
