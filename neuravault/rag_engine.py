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

class NeuraVaultRAGEngine:
    """
    Core RAG engine for NeuraVault with source tracking and citation.
    
    Attributes:
        vector_db_dir (Path): Path to persistent ChromaDB storage
        model_name (str): Ollama model to use (default: llama3.2)
        embedding_model (str): Hugging Face embedding model name
        temperature (float): LLM temperature for generation
        vector_store (Chroma): Initialized vector store
        llm (ChatOllama): Initialized language model
        qa_chain: Retrieval chain with custom source tracking
    """
    
    def __init__(
        self,
        vector_db_dir: str = "vector_db",
        model_name: str = "llama3.2",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        temperature: float = 0.3,
        chunk_k: int = 4
    ):
        """
        Initialize the NeuraVault RAG Engine.
        
        Args:
            embedding_model: Hugging Face embedding model
            temperature: LLM generation temperature (0.0-1.0)
            chunk_k: Number of document chunks to retrieve
        """
        self.vector_db_dir = Path(vector_db_dir)
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.chunk_k = chunk_k
        
        # Initialize components
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.retrieved_sources = []
        
        logger.info(
            f"Initialized NeuraVaultRAGEngine with model={model_name}, "
            f"temperature={temperature}"
        )
    
    def load_vector_store(self) -> Chroma:
        """
        Load the persistent ChromaDB vector store.
        
        Returns:
            Initialized Chroma vector store
            
        Raises:
            FileNotFoundError: If vector database doesn't exist
            Exception: If database loading fails
        """
        if not self.vector_db_dir.exists():
            logger.error(f"Vector database directory not found: {self.vector_db_dir}")
            raise FileNotFoundError(
                f"Vector database not found at {self.vector_db_dir}. "
                "Please run 'python neuravault/ingest.py' first."
            )
        
        try:
            logger.info(f"Loading vector store from {self.vector_db_dir}...")
            
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            
            vector_store = Chroma(
                persist_directory=str(self.vector_db_dir),
                embedding_function=embeddings,
                collection_name="neuravault_collection"
            )
            
            self.vector_store = vector_store
            logger.info("Vector store loaded successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}", exc_info=True)
            raise

    def initialize_llm(self) -> ChatOllama:
        """
        Initialize the Ollama LLM connection.
        
        Returns:
            Initialized ChatOllama instance
            
        Raises:
            Exception: If Ollama connection fails
        """
        try:
            logger.info(f"Initializing ChatOllama with model: {self.model_name}")
            
            llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature
            )
            
            # Test connection
            logger.info("Testing Ollama connection...")
            # Attempt a simple call to verify connection
            _ = llm.invoke("Test connection")
            
            self.llm = llm
            logger.info(f"Successfully connected to Ollama model: {self.model_name}")
            return llm
            
        except Exception as e:
            logger.error(
                f"Failed to connect to Ollama. "
                f"Ensure Ollama is running and model '{self.model_name}' is pulled. "
                f"Error: {str(e)}",
                exc_info=True
            )
            raise
            
    def create_retrieval_chain(self):
        """
        Create a retrieval chain with custom source tracking.
        
        The chain retrieves relevant documents, provides answers from the LLM,
        and includes source citations in the response.
        
        Returns:
            Configured retrieval chain
            
        Raises:
            Exception: If chain creation fails
        """
        if not self.vector_store:
            logger.error("Vector store not loaded. Call load_vector_store() first.")
            raise ValueError("Vector store must be loaded before creating chain")
        
        if not self.llm:
            logger.error("LLM not initialized. Call initialize_llm() first.")
            raise ValueError("LLM must be initialized before creating chain")
        
        try:
            logger.info("Creating retrieval chain (standard LCEL)...")

            # Define custom prompt with source citation emphasis
            prompt = ChatPromptTemplate.from_template(
                """You are a forensic analysis expert assistant for NeuraVault.
Use the following pieces of context from uploaded documents to answer the user's question.
Always cite the source documents in your response.

Context:
{context}

Question: {input}

Answer: Please provide a comprehensive answer based on the context above.
Make sure to cite which document the information comes from."""
            )

            # Create retriever with source tracking
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.chunk_k}
            )

            def _answer_with_sources(inputs: Dict[str, Any]) -> Dict[str, Any]:
                # Combine retrieved docs into a single context string for the prompt
                context_text = "\n\n".join(doc.page_content for doc in inputs["context"])
                messages = prompt.format_messages(
                    context=context_text,
                    input=inputs["input"],
                )
                llm_response = self.llm.invoke(messages)
                return {
                    "answer": getattr(llm_response, "content", llm_response),
                    "source_documents": inputs["source_documents"],
                }

            # Build retrieval chain using LCEL runnables to preserve sources
            qa_chain = (
                RunnableParallel(
                    {
                        "context": retriever,
                        "input": RunnablePassthrough(),
                        "source_documents": retriever,
                    }
                )
                | RunnableLambda(_answer_with_sources)
            )

            self.qa_chain = qa_chain
            logger.info("Retrieval chain created successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Failed to create retrieval chain: {str(e)}", exc_info=True)
            raise
            
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a user query and return answer with sources.
        
        Args:
            question: User's query string
            
        Returns:
            Dictionary containing:
                - 'answer': Generated answer from LLM
                - 'sources': List of source documents with metadata
                - 'source_texts': List of relevant text chunks used
        """
        if not self.qa_chain:
            logger.error("QA chain not initialized. Call create_retrieval_chain() first.")
            raise ValueError("QA chain must be created before querying")
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Invoke the QA chain with new format
            result = self.qa_chain.invoke(question)

            # Extract answer and sources from new format
            answer = result.get("answer", "")
            source_documents = result.get("source_documents", [])
            
            # Format sources
            sources = self._format_sources(source_documents)
            source_texts = [doc.page_content for doc in source_documents]
            
            # Store for access in UI
            self.retrieved_sources = sources
            
            logger.info(f"Query processed. Retrieved {len(sources)} source(s)")
            
            return {
                "answer": answer,
                "sources": sources,
                "source_texts": source_texts
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            raise
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents for display.
        
        Args:
            documents: List of source Document objects
            
        Returns:
            Formatted list of source information
        """
        sources = []
        for doc in documents:
            source_info = {
                "filename": doc.metadata.get("source", "Unknown"),
                "document_type": doc.metadata.get("document_type", "pdf"),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        return sources
    
