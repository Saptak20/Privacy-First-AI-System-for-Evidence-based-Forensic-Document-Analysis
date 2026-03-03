"""
test.py - Testing Module for NeuraVault

This module provides comprehensive testing functionality for NeuraVault components.
Use this to verify your installation, test individual modules, and debug issues.

Usage:
    python -m neuravault.test
    python -m neuravault.test --component ingest
    python -m neuravault.test --component rag
    python -m neuravault.test --component all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuraVaultTester:
    """
    Comprehensive testing suite for NeuraVault components.
    
    Attributes:
        data_dir (Path): Directory containing test PDFs
        vector_db_dir (Path): Vector database directory
        test_results (Dict): Storage for test results
    """
    
    def __init__(self):
        """Initialize the NeuraVault tester."""
        self.data_dir = Path("data")
        self.vector_db_dir = Path("vector_db")
        self.test_results: Dict[str, bool] = {}
        
    def print_header(self, text: str):
        """Print formatted test section header."""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)
    
    def print_test(self, name: str, passed: bool, message: str = ""):
        """
        Print test result.
        
        Args:
            name: Test name
            passed: Whether test passed
            message: Additional message
        """
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        
        print(f"{color}{status}{reset} - {name}")
        if message:
            print(f"      {message}")
        
        self.test_results[name] = passed
    
    def test_imports(self) -> bool:
        """
        Test all required package imports.
        
        Returns:
            True if all imports successful
        """
        self.print_header("Testing Package Imports")
        
        all_passed = True
        
        # Test core dependencies
        packages = [
            ("langchain", "LangChain orchestration"),
            ("langchain_community", "LangChain community integrations"),
            ("chromadb", "ChromaDB vector database"),
            ("sentence_transformers", "Sentence Transformers embeddings"),
            ("pypdf", "PyPDF document processing"),
            ("chainlit", "Chainlit web interface"),
            ("ollama", "Ollama LLM integration"),
        ]
        
        for package_name, description in packages:
            try:
                __import__(package_name)
                self.print_test(f"Import {package_name}", True, description)
            except ImportError as e:
                self.print_test(f"Import {package_name}", False, str(e))
                all_passed = False
        
        # Test NeuraVault modules
        print("\nTesting NeuraVault Modules:")
        
        modules = [
            ("neuravault.ingest", "Data ingestion pipeline"),
            ("neuravault.rag_engine", "RAG engine core"),
            ("neuravault.app", "Chainlit application"),
        ]
        
        for module_name, description in modules:
            try:
                __import__(module_name)
                self.print_test(f"Import {module_name}", True, description)
            except ImportError as e:
                self.print_test(f"Import {module_name}", False, str(e))
                all_passed = False
        
        return all_passed
    
    def test_directories(self) -> bool:
        """
        Test required directories exist.
        
        Returns:
            True if all directories exist
        """
        self.print_header("Testing Directory Structure")
        
        all_passed = True
        
        directories = [
            (self.data_dir, "PDF documents directory"),
            (self.vector_db_dir, "Vector database directory"),
            (Path("logs"), "Application logs directory"),
            (Path("neuravault"), "NeuraVault package directory"),
        ]
        
        for dir_path, description in directories:
            exists = dir_path.exists()
            self.print_test(
                f"Directory: {dir_path}",
                exists,
                description
            )
            if not exists:
                all_passed = False
        
        return all_passed
    
    def test_pdf_documents(self) -> bool:
        """
        Test for PDF documents in data directory.
        
        Returns:
            True if PDFs found
        """
        self.print_header("Testing PDF Documents")
        
        pdf_files = list(self.data_dir.glob("*.pdf"))
        count = len(pdf_files)
        
        has_pdfs = count > 0
        self.print_test(
            "PDF documents available",
            has_pdfs,
            f"Found {count} PDF file(s)"
        )
        
        if has_pdfs:
            print("\nPDF Files:")
            for pdf in pdf_files:
                size_mb = pdf.stat().st_size / (1024 * 1024)
                print(f"  - {pdf.name} ({size_mb:.2f} MB)")
        
        return has_pdfs
    
    def test_vector_database(self) -> bool:
        """
        Test vector database existence and validity.
        
        Returns:
            True if vector database exists
        """
        self.print_header("Testing Vector Database")
        
        # Check if vector_db directory exists
        db_exists = self.vector_db_dir.exists()
        self.print_test(
            "Vector DB directory exists",
            db_exists,
            str(self.vector_db_dir)
        )
        
        if not db_exists:
            return False
        
        # Check for ChromaDB files
        chroma_files = list(self.vector_db_dir.glob("*"))
        has_files = len(chroma_files) > 0
        
        self.print_test(
            "Vector DB contains files",
            has_files,
            f"Found {len(chroma_files)} file(s)"
        )
        
        if has_files:
            print("\nVector DB Contents:")
            for file in chroma_files[:10]:  # Show first 10
                print(f"  - {file.name}")
        
        return has_files
    
    def test_ollama_connection(self) -> bool:
        """
        Test Ollama service connection.
        
        Returns:
            True if Ollama is accessible
        """
        self.print_header("Testing Ollama Connection")
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                self.print_test("Ollama service running", True, "Port 11434 accessible")
                
                # Check for llama3.2 model
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                has_llama = any("llama3.2" in name for name in model_names)
                self.print_test(
                    "LLaMA 3.2 model available",
                    has_llama,
                    "Run: ollama pull llama3.2" if not has_llama else "Model ready"
                )
                
                if models:
                    print("\nAvailable Models:")
                    for model in models:
                        print(f"  - {model.get('name', 'Unknown')}")
                
                return has_llama
            else:
                self.print_test("Ollama service running", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.print_test(
                "Ollama service running",
                False,
                f"Error: {str(e)}. Run: ollama serve"
            )
            return False
    
    def test_ingestion(self) -> bool:
        """
        Test document ingestion pipeline.
        
        Returns:
            True if ingestion test passes
        """
        self.print_header("Testing Ingestion Pipeline")
        
        try:
            from neuravault.ingest import NeuraVaultIngestor
            
            # Check if PDFs exist
            pdf_count = len(list(self.data_dir.glob("*.pdf")))
            if pdf_count == 0:
                self.print_test(
                    "Ingestion test",
                    False,
                    "No PDF files in data/ directory"
                )
                return False
            
            # Initialize ingestor
            self.print_test("Initialize NeuraVaultIngestor", True)
            ingestor = NeuraVaultIngestor()
            
            # Test PDF loading
            self.print_test("Load PDF documents", True)
            docs = ingestor.load_pdf_documents()
            print(f"      Loaded {len(docs)} document(s)")
            
            # Test text cleaning
            if docs:
                sample_text = docs[0][1][:200]
                cleaned = ingestor.clean_text(sample_text)
                self.print_test("Clean text", True, f"Cleaned {len(cleaned)} characters")
            
            self.print_test("Ingestion pipeline functional", True)
            return True
            
        except Exception as e:
            self.print_test("Ingestion pipeline", False, str(e))
            logger.error("Ingestion test failed", exc_info=True)
            return False
    
    def test_rag_engine(self) -> bool:
        """
        Test RAG engine initialization and query.
        
        Returns:
            True if RAG engine test passes
        """
        self.print_header("Testing RAG Engine")
        
        try:
            from neuravault.rag_engine import NeuraVaultRAGEngine
            
            # Check vector database exists
            if not self.vector_db_dir.exists():
                self.print_test(
                    "RAG engine test",
                    False,
                    "Vector database not found. Run: python -m neuravault.ingest"
                )
                return False
            
            # Initialize RAG engine
            self.print_test("Initialize NeuraVaultRAGEngine", True)
            engine = NeuraVaultRAGEngine()
            
            # Load vector store
            self.print_test("Load vector store", True)
            engine.load_vector_store()
            
            # Initialize LLM
            self.print_test("Initialize Ollama LLM", True)
            engine.initialize_llm()
            
            # Create retrieval chain
            self.print_test("Create retrieval chain", True)
            engine.create_retrieval_chain()
            
            # Test query (optional - can be slow)
            print("\nTesting query functionality (this may take 10-30 seconds)...")
            test_question = "What is this document about?"
            
            result = engine.query(test_question)
            
            has_answer = bool(result.get("answer", ""))
            has_sources = len(result.get("sources", [])) > 0
            
            self.print_test(
                "Query processing",
                has_answer,
                f"Retrieved {len(result.get('sources', []))} source(s)"
            )
            
            if has_answer:
                print(f"\nSample Answer (first 200 chars):")
                print(f"  {result['answer'][:200]}...")
            
            return has_answer
            
        except Exception as e:
            self.print_test("RAG engine test", False, str(e))
            logger.error("RAG engine test failed", exc_info=True)
            return False
    
    def test_embedding_model(self) -> bool:
        """
        Test embedding model loading.
        
        Returns:
            True if embedding model loads successfully
        """
        self.print_header("Testing Embedding Model")
        
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            self.print_test("Import HuggingFaceEmbeddings", True)
            
            print("\nLoading sentence-transformers model (may take time on first run)...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.print_test("Load embedding model", True, "all-MiniLM-L6-v2")
            
            # Test embedding generation
            test_text = "This is a test sentence for embedding."
            embedding = embeddings.embed_query(test_text)
            
            self.print_test(
                "Generate embeddings",
                True,
                f"Vector dimension: {len(embedding)}"
            )
            
            return True
            
        except Exception as e:
            self.print_test("Embedding model test", False, str(e))
            logger.error("Embedding test failed", exc_info=True)
            return False
    
    def run_all_tests(self) -> bool:
        """
        Run complete test suite.
        
        Returns:
            True if all tests pass
        """
        print("\n" + "=" * 70)
        print("  NeuraVault Comprehensive Test Suite")
        print("=" * 70)
        
        # Run all tests
        tests = [
            ("Imports", self.test_imports),
            ("Directories", self.test_directories),
            ("PDF Documents", self.test_pdf_documents),
            ("Vector Database", self.test_vector_database),
            ("Ollama Connection", self.test_ollama_connection),
            ("Embedding Model", self.test_embedding_model),
        ]
        
        all_passed = True
        for name, test_func in tests:
            try:
                passed = test_func()
                if not passed:
                    all_passed = False
            except Exception as e:
                logger.error(f"Test '{name}' failed with exception", exc_info=True)
                all_passed = False
        
        # Print summary
        self.print_summary()
        
        return all_passed
    
    def print_summary(self):
        """Print test summary."""
        self.print_header("Test Summary")
        
        total = len(self.test_results)
        passed = sum(1 for v in self.test_results.values() if v)
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        
        if failed == 0:
            print("\n🎉 All tests passed! NeuraVault is ready to use.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Check errors above.")
            print("\nCommon Solutions:")
            print("  - Missing packages: pip install -r requirements.txt")
            print("  - No PDFs: Add PDF files to data/ folder")
            print("  - No vector DB: python -m neuravault.ingest")
            print("  - Ollama not running: ollama serve")
            print("  - Model not available: ollama pull llama3.2")


def main():
    """
    Main entry point for testing.
    
    Usage:
        python -m neuravault.test
        python -m neuravault.test --component ingest
        python -m neuravault.test --component rag
        python -m neuravault.test --component all
    """
    parser = argparse.ArgumentParser(
        description="NeuraVault Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--component",
        choices=["imports", "dirs", "pdf", "vector", "ollama", "ingest", "rag", "embedding", "all"],
        default="all",
        help="Component to test (default: all)"
    )
    
    args = parser.parse_args()
    
    tester = NeuraVaultTester()
    
    # Map component names to test functions
    component_map = {
        "imports": tester.test_imports,
        "dirs": tester.test_directories,
        "pdf": tester.test_pdf_documents,
        "vector": tester.test_vector_database,
        "ollama": tester.test_ollama_connection,
        "ingest": tester.test_ingestion,
        "rag": tester.test_rag_engine,
        "embedding": tester.test_embedding_model,
        "all": tester.run_all_tests,
    }
    
    # Run selected test
    test_func = component_map[args.component]
    
    try:
        success = test_func()
        
        if args.component != "all":
            tester.print_summary()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error("Test suite failed", exc_info=True)
        print(f"\n✗ Test suite failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
