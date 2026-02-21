"""
rag_engine.py - Retrieval-Augmented Generation Engine for NeuraVault

This module implements the core RAG logic, including:
- Loading persistent vector database
- Setting up Ollama LLM connection
- Creating retrieval chains with source tracking
- Processing queries and returning cited answers

Functions:
    - load_vector_store: Load ChromaDB from persistent storage
    - initialize_llm: Set up Ollama connection
    - create_retrieval_chain: Build RAG chain with source tracking
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
