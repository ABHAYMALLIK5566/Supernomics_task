"""
Database connection management and query execution for PostgreSQL with vector support.

Handles connection pooling, schema initialization, and provides async query execution
methods for document storage and retrieval with vector similarity search capabilities.
"""

import asyncpg
import asyncio
from typing import Optional, List, Dict, Any
from functools import lru_cache
import logging
from contextlib import asynccontextmanager

from .config import settings
from ..models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    PostgreSQL database connection manager.
    
    Handles connection pooling, query execution, and database health monitoring
    for the legal research application.
    """
    
    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """
        Initialize the database connection pool.
        
        Sets up PostgreSQL connection pool with configured parameters.
        Safe to call multiple times.
        """
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            await self._initialize_postgres()
            self._initialized = True
    
    async def _initialize_postgres(self):
        """
        Initialize PostgreSQL connection pool.
        
        Creates connection pool with vector extension support.
        """
        if self._pool is None:
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(settings.database_url)
                
                self._pool = await asyncpg.create_pool(
                    settings.database_url,
                    min_size=settings.db_pool_min_size,
                    max_size=settings.db_pool_max_size,
                    command_timeout=settings.db_command_timeout,
                    server_settings={
                        'jit': 'off'
                    }
                )
                
                async with self._pool.acquire() as conn:
                    try:
                        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    except Exception as e:
                        logger.warning(f"Could not create vector extension: {e}")
                    
                    pass
                        
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise
    
    async def close(self):
        """
        Close the database connection pool.
        
        Properly closes all connections and cleans up resources.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
        pass
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a database connection from the pool.
        
        Yields:
            asyncpg.Connection: Database connection for query execution
        """
        if self._pool is None:
            await self.initialize()
        async with self._pool.acquire() as connection:
            yield connection
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            *args: Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries
        """
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def execute_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """
        Execute a query and return single result.
        
        Args:
            query: SQL query string
            *args: Query parameters
            
        Returns:
            Optional[Dict[str, Any]]: Single result as dictionary, or None if no results
        """
        async with self.get_connection() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def execute_command(self, query: str, *args) -> str:
        """
        Execute a command query (INSERT, UPDATE, DELETE).
        
        Args:
            query: SQL command string
            *args: Query parameters
            
        Returns:
            str: Command execution status
        """
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            bool: True if database is accessible, False otherwise
        """
        try:
            async with self.get_connection() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

db_manager = DatabaseManager()

@lru_cache()
def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager: Cached database manager instance
    """
    return db_manager

async def init_database():
    """
    Initialize database schema and indexes.
    
    Creates required tables, indexes, and vector extensions for the application.
    """
    try:
        await _init_postgres_database()
        pass
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def _init_postgres_database():
    """
    Initialize PostgreSQL database schema.
    
    Creates tables, indexes, and vector extensions for document storage and retrieval.
    """
    import asyncpg
    conn = await asyncpg.connect(settings.database_url)
    
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                title VARCHAR(500),
                source VARCHAR(255),
                status VARCHAR(50) NOT NULL DEFAULT 'processed',
                metadata JSONB,
                embedding vector(1536),
                similarity_score FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        try:
            await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS status VARCHAR(50) NOT NULL DEFAULT 'processed'")
        except Exception as e:
            logger.warning(f"Could not add status column: {e}")
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                user_query TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                rag_context JSONB,
                agent_tools_used JSONB,
                response_time_ms INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        
        indexes = [
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_content_idx ON documents USING gin(to_tsvector('english', content))",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_title_idx ON documents(title)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_source_idx ON documents(source)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_metadata_idx ON documents USING gin(metadata)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_created_at_idx ON documents(created_at)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS conversation_history_session_idx ON conversation_history(session_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS conversation_history_created_at_idx ON conversation_history(created_at)"
        ]
        
        for index_query in indexes:
            try:
                await conn.execute(index_query)
            except Exception as e:
                if "already exists" not in str(e):
                    logger.warning(f"Failed to create index: {e}")
        
        try:
            await conn.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_embedding_hnsw_idx 
                ON documents USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64)
            """)
        except Exception as e:
            logger.info(f"HNSW index creation skipped (will be created after data insertion): {e}")
        
        
    finally:
        await conn.close()


async def create_hnsw_index():
    """
    Create HNSW index for vector similarity search.
    
    Should be called after inserting data for optimal performance.
    """
    try:
        async with db_manager.get_connection() as conn:
            await conn.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_embedding_hnsw_idx 
                ON documents USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64)
            """)
            pass
    except Exception as e:
        logger.error(f"Failed to create HNSW index: {e}")
        raise 