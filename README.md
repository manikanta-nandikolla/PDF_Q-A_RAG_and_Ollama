# 📄 PDF Q&A Chatbot — Local RAG with llama3.2:3b

> A fully local, privacy-first Retrieval-Augmented Generation (RAG) system that lets you chat with your PDF documents — including scanned ones — with no API keys, no cloud, and no data leaving your machine.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2:3b-black?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-persistent-orange?style=flat)
![Gradio](https://img.shields.io/badge/Gradio-6.x-FF7C00?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🖼️ Demo

> Upload PDF(s) → Ingest → Ask questions → Get cited answers with relevance scores

The app runs entirely on your local machine. No subscriptions, no rate limits, no data sent to the cloud.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Local LLM** | Powered by `llama3.2:3b` via Ollama — runs on 8 GB RAM |
| 📚 **Multi-PDF support** | Upload and query across multiple PDFs simultaneously |
| 🔍 **Semantic search** | `nomic-embed-text` embeddings + ChromaDB vector store |
| ⚡ **Parallel embedding** | Chunks embedded concurrently (4 threads) for fast ingestion |
| 🗃️ **Persistent storage** | ChromaDB saves embeddings to disk — no re-ingestion on restart |
| 🖨️ **Scanned PDF / OCR** | Automatic Tesseract OCR fallback for image-only pages |
| 💬 **Multi-turn chat** | Conversational follow-up questions with context memory |
| 📌 **Source citations** | Every answer references its source PDF filename |
| 📊 **Relevance scores** | Each retrieved chunk shows a `% match` similarity score |
| 🔎 **Source chunk panel** | Collapsible view of exact passages used to generate answers |
| ⬇️ **Export chat** | Download the full Q&A session as a formatted `.md` file |
| 🧩 **Modular codebase** | Clean separation of concerns across 5 focused modules |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        app.py                           │
│              Gradio UI — layout & event wiring          │
└────┬──────────┬──────────┬──────────┬──────────┬────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
config.py   ocr.py   embedder.py  ingestor.py  retriever.py
  All       PDF +     Ollama       Chunking +   RAG query +
 settings  Tesseract  API calls    ChromaDB     LLM answer
```

### RAG Pipeline

```
User question
      │
      ▼
[embedder.py]  Embed question via nomic-embed-text
      │
      ▼
[retriever.py] Query ChromaDB → top-3 chunks by cosine similarity
      │
      ▼
[retriever.py] Build system prompt (context + source labels)
      │
      ▼
[embedder.py]  Send to llama3.2:3b via Ollama /api/chat
      │
      ▼
Answer + source citations + relevance scores → Gradio UI
```

### Ingestion Pipeline

```
PDF file(s)
      │
      ▼
[ocr.py]       Extract text (PyMuPDF native → Tesseract OCR fallback)
      │
      ▼
[ingestor.py]  Split into 800-char chunks with 100-char overlap
      │
      ▼
[embedder.py]  Embed all chunks in parallel (ThreadPoolExecutor, 4 workers)
      │
      ▼
[ingestor.py]  Store in ChromaDB with source metadata → saved to disk
```

---

## 📁 Project Structure

```
PDF_QA_Chatbot/
│
├── app.py           # Gradio UI — layout and event bindings only
├── config.py        # All tuneable settings (models, chunk size, paths)
├── ocr.py           # PDF text extraction + Tesseract OCR fallback
├── embedder.py      # Ollama embed() and chat() API wrappers
├── ingestor.py      # Chunking, ChromaDB storage, ingest/clear handlers
├── retriever.py     # RAG query, source panel, relevance scores, export
│
├── chroma_db/       # Auto-created — persistent vector store (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|---|---|---|
| LLM | [llama3.2:3b](https://ollama.com/library/llama3.2) via Ollama | Answer generation |
| Embeddings | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) via Ollama | Semantic vector search |
| Vector store | [ChromaDB](https://www.trychroma.com/) | Persistent chunk storage & retrieval |
| PDF parsing | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | Native text extraction |
| OCR | [Tesseract](https://github.com/tesseract-ocr/tesseract) + [pytesseract](https://github.com/madmaze/pytesseract) | Scanned PDF support |
| UI | [Gradio 6](https://www.gradio.app/) | Web interface |
| Concurrency | `ThreadPoolExecutor` | Parallel chunk embedding |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-qa-chatbot.git
cd pdf-qa-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

```bash
# Download Ollama from https://ollama.com/download, then:
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama serve
```

### 5. Install Tesseract *(optional — only for scanned PDFs)*

```bash
# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Tick "Add to PATH" during install

# Linux
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

### 6. Run the app

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## 🚀 Usage

```
1. Upload   → Select one or more PDFs in the left panel
2. Ingest   → Click "Ingest PDFs" — chunks are embedded and saved to disk
3. Ask      → Type any question and press Enter or click Send
4. Explore  → Expand "Retrieved source chunks & scores" to see cited passages
5. Export   → Click "⬇ Export chat" to download the session as a .md file
6. Restart  → Embeddings persist — no re-ingestion needed next time
```

---

## ⚡ Performance

Tested on 8 GB RAM, CPU-only:

| Metric | Value |
|---|---|
| RAM usage (total) | ~2.7 GB |
| Ingestion speed | ~30s per 100-page PDF (4-thread parallel embed) |
| Answer latency | 20–60s depending on answer length |
| Max PDF size tested | 300+ pages |

**Memory breakdown:**

| Component | RAM |
|---|---|
| llama3.2:3b (q4) | ~2.0 GB |
| nomic-embed-text | ~0.3 GB |
| ChromaDB + Python | ~0.3 GB |
| Tesseract (active) | ~0.1 GB |
| **Total** | **~2.7 GB** |

---

## 🔧 Configuration

All settings live in `config.py`. Change anything here — no other file needs to be touched:

```python
# config.py

LLM_MODEL     = "llama3.2:3b"    # swap to mistral:7b for better accuracy
EMBED_MODEL   = "nomic-embed-text"

CHUNK_SIZE    = 800               # characters per chunk
CHUNK_OVERLAP = 100               # overlap between consecutive chunks
TOP_K         = 3                 # chunks retrieved per question
EMBED_WORKERS = 4                 # parallel threads — raise to 6-8 on faster CPUs

DB_PATH       = "./chroma_db"     # persistent vector store location
```

---

## 🗺️ Roadmap

- [x] Single PDF Q&A
- [x] Multi-PDF support
- [x] Scanned PDF / OCR support
- [x] Parallel embedding (4-thread)
- [x] Persistent ChromaDB vector store
- [x] Source chunk panel + relevance scores
- [x] Multi-turn chat with history
- [x] Chat history export (.md)
- [x] Modular codebase (5 focused modules)
- [ ] Page number citations
- [ ] Auto document summary on ingest
- [ ] Suggested follow-up questions
- [ ] Hybrid BM25 + vector search
- [ ] Voice input (Whisper STT) + TTS output
- [ ] Agentic multi-hop question answering

---

## 📋 requirements.txt

```
gradio>=6.0
chromadb>=0.5
PyMuPDF>=1.24
requests>=2.31
pytesseract>=0.3.10
Pillow>=10.0
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

[MIT](LICENSE)

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.com/) — local LLM inference
- [ChromaDB](https://www.trychroma.com/) — embedded vector store
- [Gradio](https://www.gradio.app/) — rapid UI framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF parsing
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — scanned document support
