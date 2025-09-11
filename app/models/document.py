"""
SQLAlchemy database models for legal documents and conversation history.

Provides database models for legal documents with vector embeddings and conversation
history tracking with proper indexing and relationship management.
"""

from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

Base = declarative_base()


class DocumentProcessingStatus(Enum):
    """Enumeration of document processing statuses"""
    PENDING = "pending"
    PROCESSED = "processed"
    ERROR = "error"


class LegalDocument(Base):
    """
    Model representing a legal document with vector embeddings for semantic search.
    
    This model stores legal documents with their content, metadata, and vector embeddings
    to enable semantic similarity search for legal research queries.
    """
    
    __tablename__ = "documents"
    
    # Primary key and content
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False, index=True)
    title = Column(String(500), nullable=True, index=True)
    source = Column(String(255), nullable=True, index=True)
    
    # Processing status
    status = Column(
        String(50), 
        nullable=False, 
        default=DocumentProcessingStatus.PROCESSED.value, 
        index=True
    )
    
    # Metadata and embeddings
    document_metadata = Column("metadata", JSON, nullable=True)
    embedding = Column(Vector(1536), nullable=True)
    similarity_score = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self) -> str:
        """String representation of the document"""
        return f"<LegalDocument(id={self.id}, title='{self.title}', source='{self.source}')>"
    
    def to_dictionary(self) -> Dict[str, Any]:
        """
        Convert the document to a dictionary for API responses.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the document with all fields
        """
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "source": self.source,
            "status": self.status,
            "metadata": self.document_metadata,
            "similarity_score": self.similarity_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def is_processed(self) -> bool:
        """
        Check if the document has been successfully processed.
        
        Returns:
            bool: True if document is processed, False otherwise
        """
        return self.status == DocumentProcessingStatus.PROCESSED.value
    
    def has_embedding(self) -> bool:
        """
        Check if the document has a vector embedding.
        
        Returns:
            bool: True if embedding exists, False otherwise
        """
        return self.embedding is not None
    
    def get_processing_status(self) -> DocumentProcessingStatus:
        """
        Get the processing status as an enum.
        
        Returns:
            DocumentProcessingStatus: The processing status enum value
        """
        try:
            return DocumentProcessingStatus(self.status)
        except ValueError:
            return DocumentProcessingStatus.ERROR

class LegalConversationHistory(Base):
    """
    Model for storing conversation history between users and the legal research assistant.
    
    This model tracks the complete conversation flow including queries, responses,
    context used, and performance metrics for analysis and improvement.
    """
    
    __tablename__ = "conversation_history"
    
    # Primary key and session tracking
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    
    # Conversation content
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    
    # Context and tools used
    rag_context = Column(JSON, nullable=True)
    agent_tools_used = Column(JSON, nullable=True)
    
    # Performance metrics
    response_time_ms = Column(Integer, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        """String representation of the conversation history entry"""
        return f"<LegalConversationHistory(id={self.id}, session_id='{self.session_id}')>"
    
    def to_dictionary(self) -> Dict[str, Any]:
        """
        Convert the conversation history to a dictionary for API responses.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the conversation history
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "rag_context": self.rag_context,
            "agent_tools_used": self.agent_tools_used,
            "response_time_ms": self.response_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def get_response_time_seconds(self) -> Optional[float]:
        """
        Get response time in seconds.
        
        Returns:
            Optional[float]: Response time in seconds, or None if not available
        """
        if self.response_time_ms is None:
            return None
        return self.response_time_ms / 1000.0
    
    def has_rag_context(self) -> bool:
        """
        Check if RAG context is available.
        
        Returns:
            bool: True if RAG context exists, False otherwise
        """
        return self.rag_context is not None and len(self.rag_context) > 0
    
    def get_tools_used_count(self) -> int:
        """
        Get the number of tools used in this conversation.
        
        Returns:
            int: Number of tools used, or 0 if none
        """
        if not self.agent_tools_used:
            return 0
        return len(self.agent_tools_used) if isinstance(self.agent_tools_used, list) else 0 