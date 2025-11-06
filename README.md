# physics-RAG-with-azure

# Physics RAG Chatbot using Azure OpenAI + FAISS

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot built to answer physics questions from your PDFs.  
It combines **Azure OpenAI embeddings**, **FAISS vector search**, and a **Flask web interface** for a seamless, local Q&A experience.

---

## Features

- **PDF Knowledge Base** — Load any physics notes or textbook (e.g., `physics_semiconductor.pdf`)  
- **Azure OpenAI Integration** — Uses Azure’s `text-embedding-3-large` for semantic search  
- **FAISS Vector Index** — Enables fast and accurate similarity search  
- **Flask Chat Interface** — Simple browser-based Q&A UI  
- **Secure Secrets Management** — Environment variables stored safely in `.env`  
- **RAG Pipeline** — Retrieval + Generation with cited sources

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| **Language Model / Embeddings** | Azure OpenAI (`text-embedding-3-large` or `text-embedding-3-small`) |
| **Vector Store** | FAISS |
| **Framework** | Flask |
| **Document Loader** | LangChain + PyMuPDF |
| **Frontend** | HTML, CSS, JS |
| **Environment Management** | Python-Dotenv |

---

## RAG Workflow

```text
        ┌─────────────┐
        │   PDF File  │
        └──────┬──────┘
               │  Split into chunks (~300 tokens)
               ▼
        ┌─────────────┐
        │ Embeddings  │  ← Azure OpenAI (text-embedding-3-large)
        └──────┬──────┘
               │  Store vectors
               ▼
        ┌─────────────┐
        │   FAISS     │  ← Vector Index
        └──────┬──────┘
               │  Retrieve top-k relevant docs
               ▼
        ┌─────────────┐
        │   Retriever │
        └──────┬──────┘
               │  Send context to LLM
               ▼
        ┌─────────────┐
        │   Chat UI   │ →  Human-like answer + sources
        └─────────────┘
