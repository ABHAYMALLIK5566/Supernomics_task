"""
Retrieval-Augmented Generation (RAG) system using OpenAI embeddings and language models.

Provides intelligent document retrieval, vector similarity search, and response generation
with caching, document ingestion, and multi-algorithm query processing capabilities.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
import numpy as np
import asyncpg

from ..core.config import settings
from ..core.database import db_manager
from ..core.exceptions import EmbeddingGenerationError, QueryProcessingError
from ..core.utils import text_processor, PerformanceTimer
from .cache import rag_cache
from .legal_tools import extract_legal_citations, classify_legal_text
from .hallucination_validator import hallucination_validator

logger = logging.getLogger(__name__)


class LightweightLLMRAG:
    """Lightweight LLM-based RAG engine with OpenAI dense embeddings and vector search."""
    
    def __init__(self):
        self.openai_client = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the RAG engine with OpenAI client and load documents."""
        if self.initialized:
            logger.info("RAG engine already initialized, forcing reinitialization...")
            self.initialized = False
            
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required but not provided")
            
            self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI client initialized successfully")
            
            # Load existing documents and verify their embedding status
            await self._load_documents_from_database()
            
            self.initialized = True
            logger.info("Lightweight LLM RAG Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    async def _load_documents_from_database(self):
        """Load documents from database and verify embeddings."""
        try:
            async with db_manager.get_connection() as conn:
                result = await conn.fetchrow("SELECT COUNT(*) as count FROM documents")
                doc_count = result['count']
                logger.info(f"Database contains {doc_count} documents")
                
                if doc_count > 0:
                    # Check how many documents have been processed with embeddings
                    result = await conn.fetchrow("SELECT COUNT(*) as count FROM documents WHERE embedding IS NOT NULL")
                    embedding_count = result['count']
                    logger.info(f"Documents with embeddings: {embedding_count}/{doc_count}")
                    
                    if embedding_count == 0:
                        logger.info("No embeddings found - documents will be processed when first queried")
                else:
                    logger.info("No documents in database - ready for ingestion")
                    
        except Exception as e:
            logger.error(f"Error loading documents from database: {e}")
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding using OpenAI embedding model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Optional[List[float]]: Generated embedding vector or None if failed
        """
        try:
            clean_text = text.strip()
            # Truncate text if it exceeds the maximum length for embedding models
            if len(clean_text) > settings.max_text_length:
                clean_text = clean_text[:settings.max_text_length]
            
            response = self.openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=clean_text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def _store_embedding(self, doc_id: int, embedding: List[float]) -> bool:
        """
        Store embedding in database.
        
        Args:
            doc_id: Document ID to store embedding for
            embedding: Embedding vector to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            async with db_manager.get_connection() as conn:
                # Convert embedding list to PostgreSQL vector format
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                
                await conn.execute(
                    "UPDATE documents SET embedding = $1::vector WHERE id = $2",
                    embedding_str, doc_id
                )
                
                logger.info(f"Stored embedding for document {doc_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing embedding for document {doc_id}: {e}")
            return False
    
    async def _get_document_embedding(self, doc_id: int) -> Optional[List[float]]:
        """
        Get document embedding from database or generate if missing.
        
        Args:
            doc_id: Document ID to get embedding for
            
        Returns:
            Optional[List[float]]: Document embedding vector or None if failed
        """
        try:
            async with db_manager.get_connection() as conn:
                result = await conn.fetchrow(
                    "SELECT embedding, content FROM documents WHERE id = $1",
                    doc_id
                )
                
                if not result:
                    return None
                
                if result['embedding'] is not None:
                    return list(result['embedding'])
                
                # Generate and store embedding if missing
                logger.info(f"Generating embedding for document {doc_id}")
                embedding = await self._generate_embedding(result['content'])
                
                if embedding:
                    await self._store_embedding(doc_id, embedding)
                    return embedding
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting embedding for document {doc_id}: {e}")
            return None
    
    async def _vector_similarity_search(self, query_embedding: List[float], top_k: int = 8, threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search for legal documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with metadata
        """
        try:
            async with db_manager.get_connection() as conn:
                # Convert query embedding to PostgreSQL vector format
                query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
                
                # Use cosine distance (<=>) for similarity search, convert to similarity score
                query = """
                SELECT 
                    id, content, title, source, metadata, 
                    1 - (embedding <=> $1::vector) as similarity_score
                FROM documents 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """
                
                results = await conn.fetch(query, query_embedding_str, top_k)
                
                relevant_docs = []
                for row in results:
                    # Only include documents above similarity threshold
                    if row['similarity_score'] >= threshold:
                        legal_citations = self._extract_legal_citations(row['content'])
                        
                        relevant_docs.append({
                            'content': row['content'],
                            'metadata': {
                                'id': row['id'],
                                'title': row['title'],
                                'source': row['source'],
                                'metadata': row['metadata'],
                                'similarity_score': float(row['similarity_score']),
                                'legal_citations': legal_citations
                            },
                            'score': float(row['similarity_score'])
                        })
                
                # Ensure results include diverse sources to prevent bias
                relevant_docs = self._ensure_source_diversity(relevant_docs)
                
                logger.info(f"Vector search found {len(relevant_docs)} relevant documents")
                return relevant_docs
                
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return []
    
    def _extract_legal_citations(self, content: str) -> List[str]:
        """Extract legal citations and article references from content"""
        import re
        
        citations = []
        
        # Regex patterns to identify common legal citation formats
        patterns = [
            r'Article\s+(\d+)',
            r'Section\s+(\d+)',
            r'Chapter\s+(\d+)',
            r'Part\s+(\d+)',
            r'(\d+)\.\s*\([a-z]\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            citations.extend(matches)
        
        # Remove duplicates and return unique citations
        return list(set(citations))
    
    def _ensure_source_diversity(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity across different document sources for comprehensive coverage"""
        if len(docs) <= 3:
            return docs
        
        # Group documents by source to prevent bias toward one source
        source_groups = {}
        for doc in docs:
            source = doc['metadata'].get('source', 'Unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        diverse_docs = []
        # Prevent one source from dominating results
        max_per_source = max(1, len(docs) // len(source_groups))
        
        for source, source_docs in source_groups.items():
            # Sort by relevance score and take top documents per source
            source_docs.sort(key=lambda x: x['score'], reverse=True)
            diverse_docs.extend(source_docs[:max_per_source])
        
        # Final sort by relevance score across all sources
        diverse_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return diverse_docs[:len(docs)]
    
    async def query(self, query: str, top_k: int = 8, use_agent: bool = False, 
                   algorithm: str = "hybrid", similarity_threshold: float = 0.25) -> Dict[str, Any]:
        """Process legal research query with RAG and return structured response."""
        start_time = time.time()
        
        try:
            # Check cache first to avoid expensive processing
            cache_key = f"query:{algorithm}:{hash(query)}"
            cached_result = await rag_cache.get_rag_query(query, algorithm)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
            
            # Check if query is too ambiguous for precise legal response
            if self._is_query_ambiguous(query):
                return {
                    "response": "Your query appears to be quite broad. For a precise legal response, please specify which particular legal provision, article, or specific aspect you'd like me to address. For example, instead of 'What is the UN Charter?', you might ask 'What does Article 41 of the UN Charter state about enforcement measures?'",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            # Generate embedding for the query to enable vector search
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return {
                    "response": "I'm unable to process your query at the moment. Please try again later.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            # Find relevant documents using vector similarity search
            relevant_docs = await self._vector_similarity_search(
                query_embedding, top_k, similarity_threshold
            )
            
            if not relevant_docs:
                return {
                    "response": "This information is not available in the provided legal documents.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            # Detect potential conflicts between documents
            conflicts = self._detect_document_conflicts(relevant_docs)
            
            # Generate response using LLM with context and conflict information
            response = await self._generate_llm_response(query, relevant_docs, conflicts)
            
            result = {
                "response": response,
                "sources": relevant_docs,
                "conflicts_detected": conflicts,
                "processing_time": time.time() - start_time
            }
            
            # Cache the result for future similar queries
            await rag_cache.cache_rag_query(query, result, algorithm)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "response": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "processing_time": time.time() - start_time
            }
    
    def _detect_document_conflicts(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect potential conflicts between documents on the same topic.
        
        Args:
            docs: List of documents to analyze for conflicts
            
        Returns:
            List[Dict[str, Any]]: List of detected conflicts
        """
        conflicts = []
        
        if len(docs) < 2:
            return conflicts
        
        # Group documents by shared legal citations to detect conflicts
        citation_groups = {}
        for doc in docs:
            citations = doc['metadata'].get('legal_citations', [])
            for citation in citations:
                if citation not in citation_groups:
                    citation_groups[citation] = []
                citation_groups[citation].append(doc)
        
        # Analyze documents that share citations for potential conflicts
        for citation, citation_docs in citation_groups.items():
            if len(citation_docs) > 1:
                conflict_indicators = self._analyze_content_conflicts(citation_docs)
                if conflict_indicators:
                    conflicts.append({
                        'citation': citation,
                        'conflicting_docs': [doc['metadata']['id'] for doc in citation_docs],
                        'conflict_indicators': conflict_indicators
                    })
        
        return conflicts
    
    def _analyze_content_conflicts(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Analyze content for potential conflicts using simple pattern matching"""
        conflict_indicators = []
        
        contradiction_patterns = [
            (r'\b(shall|must|required)\b', r'\b(may not|cannot|prohibited)\b'),
            (r'\b(prohibited|forbidden)\b', r'\b(allowed|permitted)\b'),
            (r'\b(mandatory|obligatory)\b', r'\b(optional|discretionary)\b'),
        ]
        
        content_texts = [doc['content'].lower() for doc in docs]
        
        import re
        for pos_pattern, neg_pattern in contradiction_patterns:
            for i, content in enumerate(content_texts):
                # Check for contradictory legal language within the same document
                if re.search(pos_pattern, content) and re.search(neg_pattern, content):
                    conflict_indicators.append(f"Contradictory language patterns in document {docs[i]['metadata']['id']}")
        
        return conflict_indicators
    
    def _is_query_ambiguous(self, query: str) -> bool:
        """Check if query is too broad or ambiguous for precise legal response"""
        # Only consider queries ambiguous if they are extremely short or completely unrelated to legal topics
        ambiguous_indicators = [
            len(query.split()) < 2,  # Only single word queries
            query.lower().strip() in ['what', 'how', 'why', 'when', 'where', 'who'],  # Single question words
        ]
        
        return any(ambiguous_indicators)
    
    def _generate_prompt_by_length(self, response_length: str, context: str, conflict_info: str, query: str) -> str:
        """Generate prompt based on response length setting"""
        if response_length == "short":
            return f"""You are a legal research assistant. Answer the query clearly and precisely using available documents. Include the source citation (title, article, page) at the end of your answer. Avoid asking the user for clarification. Use the provided legal documents to give a direct, specific answer. If the information is in the documents, provide it immediately. Format: Direct answer first, then specific citations from the documents.

RESPONSE REQUIREMENTS:
- Maximum 2-3 sentences
- Direct answer only
- Include key article/section references
- No background explanations

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}"""

        elif response_length == "detailed":
            return f"""You are a legal research assistant. Answer the query clearly and precisely using available documents. Include the source citation (title, article, page) at the end of your answer. Avoid asking the user for clarification. Use the provided legal documents to give a direct, specific answer. If the information is in the documents, provide it immediately. Format: Direct answer first, then specific citations from the documents.

CORE PRINCIPLES:
1. Comprehensive Answers: Provide thorough, detailed answers with complete context and background.
2. Document-Based Only: Ground all responses strictly in the provided context.
3. Complete Analysis: Include all relevant legal provisions, their relationships, and practical implications.
4. Detailed Citations: Include extensive article/section references with explanations.
5. Professional Format: Use structured formatting with clear headings and bullet points.

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}"""

        else:  # normal
            return f"""You are a legal research assistant. Answer the query clearly and precisely using available documents. Include the source citation (title, article, page) at the end of your answer. Avoid asking the user for clarification. Use the provided legal documents to give a direct, specific answer. If the information is in the documents, provide it immediately. Format: Direct answer first, then specific citations from the documents.

Context from Legal Documents:
{context}{conflict_info}

Legal Question: {query}"""

    async def _generate_llm_response(self, query: str, relevant_docs: List[Dict[str, Any]], conflicts: List[Dict[str, Any]] = None) -> str:
        """Generate response using OpenAI LLM with enhanced legal assistant guidelines"""
        try:
            context = self._prepare_structured_context(relevant_docs)
            
            conflict_info = ""
            if conflicts:
                conflict_info = "\n\nIMPORTANT - POTENTIAL CONFLICTS DETECTED:\n"
                for conflict in conflicts:
                    conflict_info += f"- Citation {conflict['citation']}: Potential conflicts between documents {conflict['conflicting_docs']}\n"
                    for indicator in conflict['conflict_indicators']:
                        conflict_info += f"  * {indicator}\n"
                conflict_info += "\nPlease address these conflicts explicitly in your response.\n"
            
            # Generate prompt based on configured response length preference
            response_length = settings.rag_response_length.lower()
            prompt = self._generate_prompt_by_length(response_length, context, conflict_info, query)
            
            # Generate response using OpenAI with low temperature for consistency
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a legal research assistant. Answer the query clearly and precisely using available documents. Include the source citation (title, article, page) at the end of your answer. Avoid asking the user for clarification. Use the provided legal documents to give a direct, specific answer. If the information is in the documents, provide it immediately. Format: Direct answer first, then specific citations from the documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Validate response to prevent hallucinated legal content
            should_reject, rejection_reason = hallucination_validator.should_reject_response(
                raw_response, context, query
            )
            
            if should_reject:
                logger.warning(f"Response rejected due to hallucination: {rejection_reason}")
                safe_response = hallucination_validator.get_safe_response(query, context)
                return safe_response
            
            # Format response for better readability
            formatted_response = self._format_response(raw_response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def _format_response(self, response: str) -> str:
        """
        Format the response to improve human readability and structure. 
        
        Args:
            response: Raw response text to format
            
        Returns:
            str: Formatted response text
        """
        if not response:
            return response
        
        formatted_response = text_processor.clean_text_comprehensive(response)
        
        import re
        
        # Clean up excessive asterisks and formatting - do this first
        formatted_response = re.sub(r'\*\*', '', formatted_response)
        formatted_response = re.sub(r'\*', '', formatted_response)
        formatted_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted_response)
        
        formatted_response = re.sub(r'\n(\d+\.)', r'\n\n\1', formatted_response)
        formatted_response = re.sub(r'\n([-\*])', r'\n\n\1', formatted_response)
        
        formatted_response = re.sub(r'\n([A-Z][A-Z\s]+:)\n', r'\n\n\1\n', formatted_response)
        formatted_response = re.sub(r'\n(\d+\.\s*[A-Z][^:]+:)\n', r'\n\n\1\n', formatted_response)
        
        formatted_response = re.sub(r'\b(Article|Section|Chapter)\s+(\d+)\b', r'\1 \2', formatted_response)
        
        formatted_response = re.sub(r'([.!?])\s*\n([A-Z])', r'\1\n\n\2', formatted_response)
        
        formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
        formatted_response = re.sub(r'^\s+', '', formatted_response, flags=re.MULTILINE)
        formatted_response = re.sub(r'\s+', ' ', formatted_response)
        formatted_response = re.sub(r'\n\s+', '\n', formatted_response)
        
        return formatted_response.strip()
    
    def _prepare_structured_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Prepare structured context with enhanced document organization and legal citations"""
        context = ""
        
        # Group documents by source for better organization
        sources = {}
        for doc in relevant_docs:
            source = doc['metadata'].get('source', 'Unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)
        
        for source, docs in sources.items():
            context += f"=== {source} ===\n"
            for i, doc in enumerate(docs, 1):
                title = doc['metadata'].get('title', f'Section {i}')
                similarity = doc['metadata'].get('similarity_score', 0)
                citations = doc['metadata'].get('legal_citations', [])
                
                # Format citations for display
                citation_text = ""
                if citations:
                    citation_text = f" [Citations: {', '.join(citations)}]"
                
                cleaned_content = self._clean_context_content(doc['content'])
                
                context += f"\n[{title}]{citation_text} (Relevance: {similarity:.2f})\n"
                context += f"{cleaned_content}\n"
            context += "\n"
        
        return context
    
    def _clean_context_content(self, content: str) -> str:
        """
        Clean content for better context presentation.
        
        Args:
            content: Raw content to clean
            
        Returns:
            str: Cleaned content
        """
        if not content:
            return content
        
        # Use utility class for comprehensive text cleaning
        return text_processor.clean_text_comprehensive(content)
    
    async def add_document(self, content: str, metadata: Dict[str, Any], 
                          source: str = "unknown", title: str = None) -> str:
        """Add a new document to the knowledge base"""
        try:
            async with db_manager.get_connection() as conn:
                import json
                # Convert metadata to JSON for storage
                metadata_json = json.dumps(metadata) if metadata else '{}'
                result = await conn.fetchrow("""
                    INSERT INTO documents (content, title, source, metadata, status)
                    VALUES ($1, $2, $3, $4::jsonb, 'pending')
                    RETURNING id
                """, content, title or "Untitled", source, metadata_json)
                
                doc_id = result['id']
                
                # Generate and store embedding for the new document
                embedding = await self._generate_embedding(content)
                if embedding:
                    await self._store_embedding(doc_id, embedding)
                    await conn.execute(
                        "UPDATE documents SET status = 'processed' WHERE id = $1",
                        doc_id
                    )
                else:
                    # Mark as error if embedding generation fails
                    await conn.execute(
                        "UPDATE documents SET status = 'error' WHERE id = $1",
                        doc_id
                    )
                
                logger.info(f"Added document {doc_id} to knowledge base")
                return str(doc_id)
                
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def add_documents_bulk(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to the knowledge base with enhanced metadata handling"""
        doc_ids = []
        
        for doc in documents:
            try:
                # Extract metadata for document processing
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'Untitled')
                source = metadata.get('source', 'bulk_import')
                
                doc_id = await self.add_document(
                    content=doc['content'],
                    metadata=metadata,
                    source=source,
                    title=title
                )
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Error adding document in bulk: {e}")
                continue
        
        logger.info(f"Successfully added {len(doc_ids)} documents in bulk")
        return doc_ids

# Global instance
lightweight_llm_rag = LightweightLLMRAG()