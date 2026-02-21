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
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text to remove noise and redundancy.
        
        Args:
            text: Raw extracted text from PDF
            
        Returns:
            Cleaned text with reduced noise
        """
        # Remove multiple consecutive whitespaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common headers/footers patterns
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive special characters while preserving readability
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\/\&]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def create_documents_with_metadata(
        self, 
        documents: List[Tuple[str, str]]
    ) -> List[Document]:
        """
        Create LangChain Document objects with metadata.
        
        Args:
            documents: List of (filename, text) tuples
            
        Returns:
            List of LangChain Document objects with metadata
        """
        doc_objects = []
        
        for filename, text in documents:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Create document with metadata
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "source": filename,
                    "document_type": "pdf"
                }
            )
            doc_objects.append(doc)
        
        logger.info(f"Created {len(doc_objects)} Document object(s) with metadata")
        return doc_objects
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(
            f"Splitting documents with chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
        
        split_docs = splitter.split_documents(documents)
        logger.info(f"Created {len(split_docs)} text chunk(s) from documents")
        
        return split_docs
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Initialize ChromaDB vector store with embeddings.
        
        Args:
            documents: List of Document chunks to embed and store
            
        Returns:
            Initialized Chroma vector store instance
        """
        logger.info(f"Initializing embeddings model: {self.model_name}")
        
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        
        logger.info(f"Creating vector store with {len(documents)} chunk(s)...")
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(self.vector_db_dir),
            collection_name="neuravault_collection"
        )
        
        logger.info(f"Vector store created and persisted to {self.vector_db_dir}")
        return vector_store
        
    def ingest(self) -> Chroma:
        """
        Main pipeline orchestration for document ingestion.
        
        Returns:
            Initialized Chroma vector store
            
        Raises:
            FileNotFoundError: If no PDF documents are found
            Exception: If any step in the pipeline fails
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting NeuraVault Document Ingestion Pipeline")
            logger.info("=" * 60)
            
            # Step 1: Load PDF documents
            raw_documents = self.load_pdf_documents()
            
            # Step 2: Create Document objects with metadata
            doc_objects = self.create_documents_with_metadata(raw_documents)
            
            # Step 3: Split documents into chunks
            split_docs = self.split_documents(doc_objects)
            
            # Step 4: Create and persist vector store
            vector_store = self.create_vector_store(split_docs)
            
            logger.info("=" * 60)
            logger.info("Ingestion pipeline completed successfully!")
            logger.info("=" * 60)
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """
    Main entry point for running the ingestion pipeline.
    
    Usage:
        python ingest.py
    """
    try:
        ingestor = NeuraVaultIngestor()
        ingestor.ingest()
        logger.info("Ingestion complete. Ready to query with rag_engine.py")
        
    except FileNotFoundError as e:
        logger.error(f"Setup Error: {str(e)}")
    except Exception as e:
        logger.error(f"Fatal Error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
