#!/usr/bin/env python3
"""
Document ingestion script for the Legal Research Assistant.
Processes all PDF files from the sample_documents/ directory.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import db_manager
from app.services.pdf_ingestion import pdf_ingestion_service


async def ingest_sample_documents():
    """Ingest all PDF documents from the sample_documents directory."""
    print("Starting document ingestion...")
    
    # Initialize database connection
    await db_manager.initialize()
    
    # Get the sample documents directory
    sample_dir = Path("sample_documents")
    if not sample_dir.exists():
        print(f"Error: {sample_dir} directory not found!")
        return
    
    # Find all PDF files
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in sample_documents directory!")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Process each PDF file
    success_count = 0
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing {pdf_file.name}...")
            
            # Ingest the PDF file
            result = await pdf_ingestion_service.ingest_single_pdf(str(pdf_file))
            
            print(f"✓ Successfully ingested {pdf_file.name}")
            print(f"  - Result: {result}")
            success_count += 1
                
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")
    
    print(f"\nIngestion complete!")
    print(f"Successfully processed: {success_count}/{len(pdf_files)} files")
    
    # Display database statistics
    try:
        async with db_manager.get_connection() as conn:
            # Count total documents
            total_docs = await conn.fetchval('SELECT COUNT(*) FROM documents')
            docs_with_embeddings = await conn.fetchval('SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL')
            
            print(f"\nDatabase Statistics:")
            print(f"  - Total documents: {total_docs}")
            print(f"  - Documents with embeddings: {docs_with_embeddings}")
            print(f"  - Embedding coverage: {docs_with_embeddings/total_docs*100:.1f}%" if total_docs > 0 else "  - No documents found")
            
    except Exception as e:
        print(f"Error retrieving database statistics: {str(e)}")


if __name__ == "__main__":
    asyncio.run(ingest_sample_documents())
