"""
Centralized configuration management with environment variable support and validation.

Provides application settings for database connections, API keys, service parameters,
and feature flags with automatic validation and type conversion.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application configuration with environment variable support.
    
    Manages all application settings including database connections, API keys,
    service parameters, and feature flags with automatic validation.
    """
    app_name: str = "RAG Microservice"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False)
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", ""),
        description="Database connection URL - REQUIRED for production"
    )
    db_pool_min_size: int = Field(default=5)
    db_pool_max_size: int = Field(default=20)
    db_command_timeout: int = Field(default=60)
    
    redis_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("REDIS_URL", None)
    )
    redis_max_connections: int = Field(default=10)
    
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key - REQUIRED for the system to function"
    )
    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
        description="OpenAI model for text generation"
    )
    openai_embedding_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        description="OpenAI model for embeddings"
    )
    openai_assistant_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_ASSISTANT_ID", None),
        description="OpenAI Assistant ID (optional)"
    )
    
    complex_query_keywords: List[str] = Field(
        default=[
            "analyze", "compare", "evaluate", "explain why", "logic", 
            "step by step", "calculate", "prove", "solve", "optimize", "strategy",
            "complex", "detailed analysis", "comprehensive", "elaborate", "justify"
        ]
    )
    
    rag_top_k: int = Field(default=5)
    rag_similarity_threshold: float = Field(default=0.7)
    rag_max_tokens: int = Field(default=4000)
    rag_response_length: str = Field(
        default_factory=lambda: os.getenv("RAG_RESPONSE_LENGTH", "normal"),
        description="Response length: 'short', 'normal', or 'detailed'"
    )
    
    cache_ttl_seconds: int = Field(default=300)
    cache_max_query_length: int = Field(default=1000)
    
    
    cors_origins: str = Field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS", 
            "http://localhost:3001,http://localhost:3002,http://localhost:80,http://localhost"
        )
    )
    
    rate_limit_requests: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
    )
    rate_limit_window: str = Field(
        default_factory=lambda: os.getenv("RATE_LIMIT_WINDOW", "1/minute")
    )
    
    websocket_max_connections: int = Field(default=1000)
    websocket_ping_interval: int = Field(default=30)
    
    enable_metrics: bool = Field(default=True)
    log_level: str = Field(default="DEBUG")
    
    max_request_size_mb: int = Field(default=100)
    allowed_file_types: str = Field(default="pdf,txt,docx,md")
    
    # Text processing limits
    max_text_length: int = Field(default=8000, description="Maximum text length for processing")
    pdf_chunk_size: int = Field(default=1000, description="PDF chunk size for processing")
    pdf_chunk_overlap: int = Field(default=200, description="PDF chunk overlap for processing")
    
    test_mode: bool = Field(default=False)
    disable_database_init: bool = Field(default=False)
    disable_cache_init: bool = Field(default=False)
    
    
    @field_validator('redis_url')
    @classmethod
    def validate_redis_url(cls, v):
        if v and v.strip() == "":
            return None
        return v
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable.")
        return v
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Database URL is required. Please set DATABASE_URL environment variable.")
        if v.startswith("sqlite://"):
            return v
        if not v.startswith("postgresql://") and not v.startswith("postgres://"):
            raise ValueError("Only PostgreSQL and SQLite databases are supported")
        return v
    
    
    @field_validator('max_text_length')
    @classmethod
    def validate_max_text_length(cls, v):
        if v < 1000:
            raise ValueError("max_text_length must be at least 1000 characters")
        if v > 50000:
            raise ValueError("max_text_length should not exceed 50000 characters for performance")
        return v
    
    @field_validator('pdf_chunk_size')
    @classmethod
    def validate_pdf_chunk_size(cls, v):
        if v < 100:
            raise ValueError("pdf_chunk_size must be at least 100 characters")
        if v > 10000:
            raise ValueError("pdf_chunk_size should not exceed 10000 characters")
        return v
    
    model_config = ConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False
    )

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Cached application settings instance
    """
    return Settings()

settings = get_settings() 