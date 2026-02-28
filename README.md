# NeuraVault — Privacy-First RAG Application (Next.js + FastAPI)

**NeuraVault** is a modular, offline-first Retrieval-Augmented Generation (RAG) system designed for forensic document analysis. It allows users to upload PDF documents, ask questions, and receive answers with strict source citations—all while maintaining 100 % data privacy through local processing.

## Project Overview

### What is NeuraVault?

NeuraVault is a forensic analysis system that combines:
- **Local LLM Processing**: Uses Ollama with LLaMA 3.2 for on-device inference
- **Vector Embeddings**: Hugging Face Sentence Transformers for semantic search
- **Persistent Storage**: ChromaDB for fast document retrieval
- **Modern UI**: Next.js web interface with Tailwind CSS
- **REST API**: FastAPI backend for clean separation of frontend and backend
- **Source Citations**: Every answer includes exact document references

### Key Features

**Privacy-First**: 100% offline operation — no data leaves your computer  
**Modular Architecture**: Clean separation of concerns across multiple modules  
**Source Tracking**: Responses include document filenames and relevant excerpts  
**Persistent Storage**: Vector embeddings cached for fast subsequent queries  
**Error Handling**: Comprehensive error checks and informative messages  
**Logging**: Detailed logs for debugging and auditing  
**Type Hints**: Full type annotations for code clarity  

## Project Structure

```
Mini_Project_NV/
├── requirements.txt              # Python dependencies
├── neuravault/                   # Python backend package
│   ├── __init__.py              # Package initialization
│   ├── api.py                   # FastAPI REST endpoints
│   ├── ingest.py                # Data pipeline (PDF → Vector DB)
│   ├── rag_engine.py            # RAG logic & LLM orchestration
│   └── test.py                  # Testing suite
├── frontend/                    # Next.js web application
│   ├── package.json
│   ├── next.config.js           # Proxies /api/* to FastAPI
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── app/                 # App Router (layout, page, globals)
│   │   ├── components/          # ChatWindow, MessageBubble, etc.
│   │   └── lib/api.ts           # API client helpers
├── data/                        # Input PDF documents (user-provided)
├── vector_db/                   # ChromaDB persistent storage
├── logs/                        # Application logs
└── scripts/
    └── make_fake_data.py        # Generate test PDFs
```

### File Descriptions

#### [requirements.txt](requirements.txt)
Contains all Python package dependencies:
- **langchain** — LLM orchestration framework
- **chromadb** — Vector database for embeddings
- **sentence-transformers** — Embedding model
- **pypdf** — PDF text extraction
- **fastapi** + **uvicorn** — REST API backend
- **ollama** — Local LLM integration

#### [neuravault/ingest.py](neuravault/ingest.py)
**Purpose**: Data pipeline for document ingestion

**Key Classes**:
- `NeuraVaultIngestor`: Manages PDF loading and embedding generation

**Key Functions**:
- `load_pdf_documents()` - Extract text from PDFs with page tracking
- `clean_text()` - Remove noise and formatting artifacts
- `split_documents()` - Chunk text using RecursiveCharacterTextSplitter
- `create_vector_store()` - Initialize ChromaDB with embeddings
- `ingest()` - Orchestrate the complete pipeline

**Workflow**:
1. Load all PDFs from `data/` directory
2. Extract and clean text with metadata preservation
3. Split into 500-character chunks (50-char overlap)
4. Generate Sentence Transformer embeddings
5. Persist vectors in ChromaDB for reuse

#### [neuravault/rag_engine.py](neuravault/rag_engine.py)
**Purpose**: Core RAG engine with LLM integration

**Key Classes**:
- `NeuraVaultRAGEngine`: Manages retrieval and generation

**Key Functions**:
- `load_vector_store()` - Load persisted ChromaDB
- `initialize_llm()` - Connect to Ollama LLaMA 3.2
- `create_retrieval_chain()` - Build RetrievalQA with custom prompt
- `query()` - Process user questions and return cited answers

**Architecture**:
- Loads ChromaDB from persistent storage
- Connects to Ollama (must be running)
- Creates RetrievalQA chain with:
  - Cosine similarity retrieval (k=4 chunks)
  - Custom prompt emphasizing source citations
  - Returns both answer AND source documents

#### [neuravault/api.py](neuravault/api.py)
**Purpose**: FastAPI REST backend (replaces Chainlit)

**Key Endpoints**:
- `GET /api/health` — Health / readiness check
- `GET /api/starters` — Suggested starter questions
- `POST /api/query` — Send a question, receive answer with sources
- `POST /api/upload` — Upload PDF files to data/
- `POST /api/ingest` — Run the ingestion pipeline

#### [frontend/](frontend/)
**Purpose**: Next.js 14 web interface with Tailwind CSS

**Key Components**:
- `ChatWindow` — Main chat container with message list and input bar
- `MessageBubble` — Renders user/assistant messages with Markdown
- `SourceCard` — Expandable source citation card
- `StarterQuestions` — Clickable starter question grid
- `Sidebar` — System status and information panel

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Ollama installed and running
- 4GB+ RAM recommended
- Internet connection (for initial model download only)

### Step 1: Setup Python Environment

```bash
cd Mini_Project_NV
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

### Step 2: Set Up Ollama

```bash
ollama pull llama3.2
ollama serve                    # keep running in a separate terminal
```

### Step 3: Prepare & Ingest Documents

Place PDFs in `data/`, or generate test data:

```bash
python scripts/make_fake_data.py
python -m neuravault.ingest
```

### Step 4: Start the FastAPI Backend

```bash
uvicorn neuravault.api:app --reload --port 8000
```

### Step 5: Start the Next.js Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

## Usage Examples

### Example 1: Basic Query

**Question**: "What are the main topics covered?"

**Expected Response**:
```
The documents discuss forensic analysis, digital evidence 
handling, and chain of custody procedures...

Sources
Source 1: document1.pdf
Type: pdf
Relevant Excerpt: Forensic analysis is the process of 
examining evidence...
```

### Example 2: Specific Information

**Question**: "What procedures should be followed for evidence collection?"

**Expected Response**:
```
According to the documents, the following procedures should 
be followed:
1. Establish chain of custody...
2. Use proper containers...
3. Document all findings...

Sources
Source 1: document2.pdf
Type: pdf
Relevant Excerpt: Chain of custody procedures...
```

## Configuration

### Customizing RAG Parameters

Edit `neuravault/rag_engine.py`:

```python
engine = NeuraVaultRAGEngine(
    vector_db_dir="vector_db",      # Vector DB location
    model_name="llama3.2",          # Ollama model to use
    temperature=0.3,                # LLM creativity (0.0=focused, 1.0=creative)
    chunk_k=4                       # Number of chunks to retrieve
)
```

### Customizing Ingestion Parameters

Edit `neuravault/ingest.py`:

```python
ingestor = NeuraVaultIngestor(
    data_dir="data",                # PDF directory
    vector_db_dir="vector_db",      # Embedding storage
    chunk_size=500,                 # Characters per chunk
    chunk_overlap=50,               # Overlap between chunks
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

## Architecture Diagram

```
User Query
    ↓
Next.js Frontend (localhost:3000)
    ↓  HTTP /api/query
FastAPI Backend  (localhost:8000)
    ↓
RAG Engine (rag_engine.py)
    ├─→ Vector Store (ChromaDB) ←─ Embeddings (Hugging Face)
    │       ↓
    │   Retrieve Relevant Chunks (k=4)
    │       ↓
    └─→ LLM (Ollama/LLaMA 3.2)
            ↓
        Generate Answer with Sources
            ↓
        JSON Response → Display in UI
```

## Debugging

### Check Ollama Connection

```bash
# Test Ollama is running
ollama list

# Should show llama3.2 in list
# If not, run: ollama pull llama3.2
```

### View Application Logs

```bash
# Ingestion logs
cat logs/ingest.log

# RAG engine logs
cat logs/rag_engine.log

# API backend logs
cat logs/api.log
```

### Verify Vector Database

```bash
# Check vector_db directory exists
ls -la vector_db/

# Should contain ChromaDB files
```

## Logging

NeuraVault generates detailed logs in the `logs/` directory:

- **ingest.log** - PDF loading, text cleaning, and embedding generation
- **rag_engine.log** - Query processing and Ollama interactions
- **api.log** - FastAPI backend events and request handling

Each log includes timestamps and severity levels (INFO, WARNING, ERROR).

## Privacy & Security

### Data Privacy

**100% Offline Processing**
- No data is sent to external servers
- LLM runs locally on your computer
- Embeddings stored locally in ChromaDB

**No Cloud Dependencies**
- Ollama runs locally
- Sentence Transformers downloaded once, cached locally
- Vector database persists on disk

**Data Retention**
- PDFs stored in `data/` folder only
- Vectors persisted in `vector_db/` for fast retrieval
- Remove either directory to clear data

## Troubleshooting

### "No PDF files found in data/"

**Solution**:
```bash
# Verify data directory exists
ls data/

# Add PDF files to data/ folder
# Then run ingestion again
python -m neuravault.ingest
```

### "Failed to connect to Ollama"

**Solution**:
```bash
# Start Ollama in separate terminal
ollama serve

# Verify model is pulled
ollama list
ollama pull llama3.2

# Then restart Chainlit
```

### "Vector database not found"

**Solution**:
```bash
# Run ingestion pipeline first
python -m neuravault.ingest

# Verify vector_db/ directory created
ls vector_db/
```

### "Module not found" errors

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify you're in virtual environment
which python  # Should show venv/bin/python
```

## Code Examples

### Using NeuraVaultIngestor Directly

```python
from neuravault.ingest import NeuraVaultIngestor

ingestor = NeuraVaultIngestor()
vector_store = ingestor.ingest()
```

### Using NeuraVaultRAGEngine Directly

```python
from neuravault.rag_engine import initialize_rag_engine

engine = initialize_rag_engine()
result = engine.query("What are the key findings?")

print("Answer:", result["answer"])
print("Sources:", result["sources"])
```

## Additional Resources

- [LangChain Documentation](https://docs.langchain.com)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Next.js Documentation](https://nextjs.org/docs)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Hugging Face Sentence Transformers](https://www.sbert.net/)

## Contributing

Contributions welcome! Areas for enhancement:
- Support for additional document formats (DOCX, TXT)
- Multiple language support
- Advanced filtering and search options
- Performance optimizations
- Additional embedding models

## License

This project is provided as-is for educational and research purposes.

## Technical Specifications

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10+ |
| LLM | LLaMA | 3.2 |
| LLM Framework | Ollama | Latest |
| Embeddings | Sentence Transformers | 2.2.2 |
| Vector DB | ChromaDB | 0.4.24 |
| Orchestration | LangChain | 0.1.17 |
| Backend API | FastAPI | Latest |
| Frontend | Next.js | 14 |
| PDF Processing | PyPDF | 4.0.0 |

