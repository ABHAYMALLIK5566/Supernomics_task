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

logger = logging.getLogger(__name__)


class LightweightLLMRAG:
    """Lightweight LLM-based RAG engine with OpenAI dense embeddings"""
    
    def __init__(self):
        self.openai_client = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the RAG engine with OpenAI client"""
        if self.initialized:
            logger.info("RAG engine already initialized, forcing reinitialization...")
            self.initialized = False
            
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required but not provided")
            
            self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            logger.info("✅ OpenAI client initialized successfully")
            
            await self._load_documents_from_database()
            
            self.initialized = True
            logger.info("✅ Lightweight LLM RAG Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG engine: {e}")
            raise
    
    async def _load_documents_from_database(self):
        """Load documents from database and verify embeddings"""
        try:
            async with db_manager.get_connection() as conn:
                result = await conn.fetchrow("SELECT COUNT(*) as count FROM documents")
                doc_count = result['count']
                logger.info(f"Database contains {doc_count} documents")
                
                if doc_count > 0:
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
        """Generate embedding using OpenAI embedding model"""
        try:
            clean_text = text.strip()
            if len(clean_text) > 8000:
                clean_text = clean_text[:8000]
            
            response = self.openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=clean_text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def _store_embedding(self, doc_id: int, embedding: List[float]) -> bool:
        """Store embedding in database"""
        try:
            async with db_manager.get_connection() as conn:
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
        """Get document embedding from database or generate if missing"""
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
        """Perform enhanced vector similarity search with better legal document retrieval"""
        try:
            async with db_manager.get_connection() as conn:
                query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
                
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
        
        return list(set(citations))
    
    def _ensure_source_diversity(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure diversity across different document sources for comprehensive coverage"""
        if len(docs) <= 3:
            return docs
        
        source_groups = {}
        for doc in docs:
            source = doc['metadata'].get('source', 'Unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        diverse_docs = []
        max_per_source = max(1, len(docs) // len(source_groups))
        
        for source, source_docs in source_groups.items():
            source_docs.sort(key=lambda x: x['score'], reverse=True)
            diverse_docs.extend(source_docs[:max_per_source])
        
        diverse_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return diverse_docs[:len(docs)]
    
    async def query(self, query: str, top_k: int = 8, use_agent: bool = False, 
                   algorithm: str = "hybrid", similarity_threshold: float = 0.25) -> Dict[str, Any]:
        """Enhanced query method with conflict detection and comprehensive legal analysis"""
        start_time = time.time()
        
        try:
            cache_key = f"query:{algorithm}:{hash(query)}"
            cached_result = await rag_cache.get_rag_query(query, algorithm)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
            
            if self._is_query_ambiguous(query):
                return {
                    "response": "Your query appears to be quite broad. For a precise legal response, please specify which particular legal provision, article, or specific aspect you'd like me to address. For example, instead of 'What is the UN Charter?', you might ask 'What does Article 41 of the UN Charter state about enforcement measures?'",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return {
                    "response": "I'm unable to process your query at the moment. Please try again later.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            relevant_docs = await self._vector_similarity_search(
                query_embedding, top_k, similarity_threshold
            )
            
            if not relevant_docs:
                return {
                    "response": "This information is not available in the provided legal documents.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            conflicts = self._detect_document_conflicts(relevant_docs)
            
            response = await self._generate_llm_response(query, relevant_docs, conflicts)
            
            result = {
                "response": response,
                "sources": relevant_docs,
                "conflicts_detected": conflicts,
                "processing_time": time.time() - start_time
            }
            
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
        """Detect potential conflicts between documents on the same topic"""
        conflicts = []
        
        if len(docs) < 2:
            return conflicts
        
        citation_groups = {}
        for doc in docs:
            citations = doc['metadata'].get('legal_citations', [])
            for citation in citations:
                if citation not in citation_groups:
                    citation_groups[citation] = []
                citation_groups[citation].append(doc)
        
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
        
        for pos_pattern, neg_pattern in contradiction_patterns:
            import re
            for i, content in enumerate(content_texts):
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
            
            prompt = f"""You are a legal research assistant focused on delivering direct, actionable information based strictly on the provided legal documents. Generate responses that are concise, practical, and immediately useful.

                CORE PRINCIPLES:
                1. Direct Answers: Provide immediate, actionable answers to the specific question asked. For overview questions (like "What is Article 41?"), provide a comprehensive overview with key provisions and practical implications.
                2. Document-Based Only: Ground all responses strictly in the provided context. If information is missing, state "This information is not available in the provided legal documents."
                3. Combine Related Information: When a question references multiple related legal provisions (e.g., Articles 41, 42, and 51 of the UN Charter), synthesize and integrate content from all relevant document chunks to explain their legal relationship or interaction. Avoid answering based on partial or isolated excerpts.
                4. Actionable Information: Focus on what the user can do, what applies, or what the legal provisions mean in practice.
                5. Concise Format: Use bullet points, numbered lists, and clear headings to make information quickly scannable.
                6. Specific Citations: Include exact article/section references for immediate verification.
                
                FORMATTING RULES:
                - Use simple bullet points with dashes (-) only
                - Reference articles as "Article 41" without asterisks or special formatting
                - Use clean paragraph breaks
                - Avoid bold text, asterisks, or special characters
                - Keep formatting minimal and professional

                RESPONSE STRUCTURE:
                - Lead with the direct answer to the question
                - For overview questions, provide a structured overview with key provisions
                - Provide specific legal provisions with exact citations
                - Include practical implications or requirements
                - Use clean bullet points for multiple related points
                - End with key takeaways or next steps if applicable

                AVOID:
                - Lengthy explanations of legal theory
                - General background information not directly relevant
                - Repetitive or redundant content
                - Overly complex legal jargon without practical context
                - Any use of asterisks (*), bold formatting, or special characters
                - Multiple consecutive newlines or excessive whitespace

                Context from Legal Documents:
                {context}{conflict_info}

                Legal Question: {query}

                IMPORTANT: If the question asks to "define" a term, treat it the same as "explain" - provide a comprehensive explanation based on the document content, not a formal dictionary definition.

                Legal Answer:"""
            
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a legal research assistant focused on delivering direct, actionable information. Provide concise, practical responses that immediately answer the user's question with specific legal provisions and clear citations. Use clean, professional formatting with simple bullet points and no asterisks, bold text, or special characters. Reference articles as 'Article 41' without any formatting. Focus on what the user needs to know to take action."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            raw_response = response.choices[0].message.content.strip()
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
        
        # Use utility class for basic text cleaning
        formatted_response = text_processor.clean_text_comprehensive(response)
        
        # Apply additional formatting specific to responses
        import re
        
        # Clean up excessive asterisks and formatting - do this first
        formatted_response = re.sub(r'\*\*', '', formatted_response)  # Remove double asterisks first
        formatted_response = re.sub(r'\*', '', formatted_response)  # Remove single asterisks
        formatted_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted_response)  # Clean up excessive newlines
        
        # Format lists and sections with clean bullet points
        formatted_response = re.sub(r'\n(\d+\.)', r'\n\n\1', formatted_response)
        formatted_response = re.sub(r'\n([•\-\*])', r'\n\n\1', formatted_response)
        
        # Format headings without excessive asterisks
        formatted_response = re.sub(r'\n([A-Z][A-Z\s]+:)\n', r'\n\n\1\n', formatted_response)
        formatted_response = re.sub(r'\n(\d+\.\s*[A-Z][^:]+:)\n', r'\n\n\1\n', formatted_response)
        
        # Highlight legal references without asterisks
        formatted_response = re.sub(r'\b(Article|Section|Chapter)\s+(\d+)\b', r'\1 \2', formatted_response)
        
        # Improve paragraph spacing
        formatted_response = re.sub(r'([.!?])\s*\n([A-Z])', r'\1\n\n\2', formatted_response)
        
        # Final cleanup - remove excessive whitespace and newlines
        formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
        formatted_response = re.sub(r'^\s+', '', formatted_response, flags=re.MULTILINE)
        formatted_response = re.sub(r'\s+', ' ', formatted_response)  # Replace multiple spaces with single space
        formatted_response = re.sub(r'\n\s+', '\n', formatted_response)  # Remove leading spaces from lines
        
        return formatted_response.strip()
    
    def _prepare_structured_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Prepare structured context with enhanced document organization and legal citations"""
        context = ""
        
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
                metadata_json = json.dumps(metadata) if metadata else '{}'
                result = await conn.fetchrow("""
                    INSERT INTO documents (content, title, source, metadata, status)
                    VALUES ($1, $2, $3, $4::jsonb, 'pending')
                    RETURNING id
                """, content, title or "Untitled", source, metadata_json)
                
                doc_id = result['id']
                
                embedding = await self._generate_embedding(content)
                if embedding:
                    await self._store_embedding(doc_id, embedding)
                    await conn.execute(
                        "UPDATE documents SET status = 'processed' WHERE id = $1",
                        doc_id
                    )
                else:
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
        """Add multiple documents to the knowledge base"""
        doc_ids = []
        
        for doc in documents:
            try:
                doc_id = await self.add_document(
                    content=doc['content'],
                    metadata=doc.get('metadata', {}),
                    source=doc.get('source', 'bulk_import'),
                    title=doc.get('title')
                )
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Error adding document in bulk: {e}")
                continue
        
        logger.info(f"Successfully added {len(doc_ids)} documents in bulk")
        return doc_ids

# Global instance
lightweight_llm_rag = LightweightLLMRAG()