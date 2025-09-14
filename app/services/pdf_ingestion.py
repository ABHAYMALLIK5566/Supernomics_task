"""
PDF document ingestion service for text extraction and knowledge base integration.

Provides PDF text extraction, intelligent chunking, and document ingestion
into the legal knowledge base with error handling and metadata management.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from PyPDF2 import PdfReader
from ..core.exceptions import DocumentProcessingError
from ..core.utils import text_processor
from ..core.config import settings
from .lightweight_llm_rag import lightweight_llm_rag
from .enhanced_metadata_processor import metadata_processor

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Extracts and cleans text from PDF documents with error handling."""
    
    def __init__(self):
        self.max_text_length = settings.max_text_length
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file with enhanced cleaning.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted and cleaned text
            
        Raises:
            DocumentProcessingError: If PDF extraction fails
        """
        try:
            if not os.path.exists(pdf_path):
                raise DocumentProcessingError(
                    f"PDF file not found: {pdf_path}",
                    document_id=os.path.basename(pdf_path),
                    processing_stage="file_validation"
                )
            
            reader = PdfReader(pdf_path)
            extracted_text = ""
            
            for page_number, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_page_text = self._clean_extracted_text(page_text)
                        if cleaned_page_text.strip():
                            extracted_text += f"\n--- Page {page_number} ---\n"
                            extracted_text += cleaned_page_text
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_number}: {e}")
                    continue
            
            if not extracted_text.strip():
                raise DocumentProcessingError(
                    f"No text extracted from PDF: {pdf_path}",
                    document_id=os.path.basename(pdf_path),
                    processing_stage="text_extraction"
                )
            
            return self._clean_extracted_text(extracted_text)
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to read PDF {pdf_path}: {str(e)}",
                document_id=os.path.basename(pdf_path),
                processing_stage="pdf_reading"
            )
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text to remove formatting artifacts and improve readability.
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return text
        
        # Use utility class for comprehensive text cleaning
        return text_processor.clean_text_comprehensive(text)


class TextChunker:
    """Splits text into manageable chunks while preserving sentence boundaries."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk (defaults to config)
            chunk_overlap: Overlap between consecutive chunks (defaults to config)
        """
        self.chunk_size = chunk_size or settings.pdf_chunk_size
        self.chunk_overlap = chunk_overlap or settings.pdf_chunk_overlap
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks while preserving sentence boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start_position = 0
        
        while start_position < len(text):
            end_position = start_position + self.chunk_size
            
            if end_position < len(text):
                # Try to break at sentence boundary
                end_position = self._find_sentence_boundary(text, start_position, end_position)
            
            chunk = text[start_position:end_position].strip()
            if chunk:
                chunks.append(chunk)
            
            start_position = end_position - self.chunk_overlap
            if start_position >= len(text):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find the best sentence boundary within the given range.
        
        Args:
            text: Full text
            start: Start position
            end: End position
            
        Returns:
            int: Best boundary position
        """
        # Look for sentence endings
        for i in range(end, max(start + self.chunk_size - 100, start), -1):
            if text[i] in '.!?':
                return i + 1
        
        # Fall back to line breaks
        for i in range(end, max(start + self.chunk_size - 100, start), -1):
            if text[i] == '\n':
                return i
        
        return end


class PDFIngestionService:
    """Service for ingesting PDF documents into the legal knowledge base."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the PDF ingestion service.
        
        Args:
            chunk_size: Size of text chunks for processing (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
        """
        self.text_extractor = PDFTextExtractor()
        self.text_chunker = TextChunker(chunk_size, chunk_overlap)
    
    async def ingest_single_pdf(self, pdf_path: str, source: str = "pdf") -> str:
        """
        Ingest a single PDF file into the legal knowledge base.
        
        Args:
            pdf_path: Path to the PDF file
            source: Source identifier for the document
            
        Returns:
            str: Success message with number of chunks ingested
            
        Raises:
            DocumentProcessingError: If PDF ingestion fails
        """
        try:
            # Extract text from PDF
            extracted_text = self.text_extractor.extract_text_from_pdf(pdf_path)
            
            # Split text into chunks
            text_chunks = self.text_chunker.split_text_into_chunks(extracted_text)
            
            # Prepare documents for ingestion
            documents = self._prepare_documents_for_ingestion(
                text_chunks, pdf_path, source
            )
            
            # Add documents to knowledge base
            document_ids = await lightweight_llm_rag.add_documents_bulk(documents)
            
            logger.info(f"Successfully ingested PDF {pdf_path} with {len(document_ids)} chunks")
            return f"Successfully ingested {len(document_ids)} chunks from {os.path.basename(pdf_path)}"
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to ingest PDF {pdf_path}: {str(e)}",
                document_id=os.path.basename(pdf_path),
                processing_stage="ingestion"
            )
    
    async def ingest_multiple_pdfs(self, pdf_paths: List[str], source: str = "pdf") -> List[str]:
        """
        Ingest multiple PDF files into the legal knowledge base.
        
        Args:
            pdf_paths: List of paths to PDF files
            source: Source identifier for the documents
            
        Returns:
            List[str]: List of results for each PDF file
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = await self.ingest_single_pdf(pdf_path, source)
                results.append(result)
            except DocumentProcessingError as e:
                error_message = f"Failed to ingest {pdf_path}: {e.message}"
                logger.error(error_message)
                results.append(error_message)
            except Exception as e:
                error_message = f"Failed to ingest {pdf_path}: {str(e)}"
                logger.error(error_message)
                results.append(error_message)
        
        return results
    
    def _prepare_documents_for_ingestion(
        self, 
        text_chunks: List[str], 
        pdf_path: str, 
        source: str
    ) -> List[Dict[str, Any]]:
        """
        Prepare document data for ingestion into the knowledge base with enhanced metadata.
        
        Args:
            text_chunks: List of text chunks from the PDF
            pdf_path: Path to the original PDF file
            source: Source identifier
            
        Returns:
            List[Dict[str, Any]]: List of document data dictionaries with enhanced metadata
        """
        documents = []
        filename = os.path.basename(pdf_path)
        
        for chunk_index, chunk in enumerate(text_chunks):
            if chunk.strip():
                # Process enhanced metadata
                enhanced_metadata = metadata_processor.process_document_metadata(
                    content=chunk.strip(),
                    filename=filename,
                    chunk_index=chunk_index,
                    total_chunks=len(text_chunks),
                    file_path=pdf_path,
                    source=source
                )
                
                document_data = {
                    "content": chunk.strip(),
                    "metadata": enhanced_metadata
                }
                documents.append(document_data)
        
        return documents


# Global service instance
pdf_ingestion_service = PDFIngestionService()


# Backward compatibility functions
async def ingest_pdfs(pdf_paths: List[str], source: str = "pdf") -> List[str]:
    """
    Backward compatibility function for PDF ingestion.
    
    Args:
        pdf_paths: List of PDF file paths
        source: Source identifier
        
    Returns:
        List[str]: Results of ingestion for each PDF
    """
    return await pdf_ingestion_service.ingest_multiple_pdfs(pdf_paths, source)
