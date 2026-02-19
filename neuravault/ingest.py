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
