"""
ingest.py - Data Pipeline Module for NeuraVault

This module handles the ingestion of PDF documents into the vector database.
It includes text cleaning, chunking, and embedding generation.

Functions:
    - load_pdf_documents: Load all PDFs from the data directory
    - clean_text: Remove noise and redundancy from extracted text
    - create_vector_store: Initialize ChromaDB with embeddings
    - ingest_documents: Main pipeline orchestration
"""

import os
import logging
from typing import List, Tuple
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Creating class NeuraVaultIngestor
class NeuraVaultIngestor:
    """
    Manages the ingestion of documents into the NeuraVault vector store.
    
    Attributes:
        data_dir (Path): Directory containing PDF documents
        vector_db_dir (Path): Directory for persistent vector database storage
        chunk_size (int): Size of text chunks for splitting
        chunk_overlap (int): Overlap between consecutive chunks
        model_name (str): Hugging Face embedding model identifier
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        vector_db_dir: str = "vector_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the NeuraVault Ingestor.
        
        Args:
            data_dir: Path to directory containing PDF files
            vector_db_dir: Path to vector database directory
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            model_name: Hugging Face embedding model name
        """
        self.data_dir = Path(data_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized NeuraVaultIngestor with data_dir={data_dir}")
    
    def load_pdf_documents(self) -> List[Tuple[str, str]]:
        """
        Load all PDF documents from the data directory.
        
        Returns:
            List of tuples containing (filename, text content)
            
        Raises:
            FileNotFoundError: If no PDF files are found in the data directory
        """
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_dir}")
            raise FileNotFoundError(
                f"No PDF documents found in '{self.data_dir}'. "
                "Please add PDF files to the data/ directory."
            )
        
        documents = []
        logger.info(f"Found {len(pdf_files)} PDF file(s). Starting extraction...")
        
        for pdf_path in pdf_files:
            try:
                logger.info(f"Processing: {pdf_path.name}")
                reader = PdfReader(pdf_path)
                text = ""
                
                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}"
                
                if text.strip():
                    documents.append((pdf_path.name, text))
                    logger.info(f"Extracted {len(reader.pages)} pages from {pdf_path.name}")
                else:
                    logger.warning(f"No text extracted from {pdf_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                raise
        
        logger.info(f"Successfully loaded {len(documents)} document(s)")
        return documents
