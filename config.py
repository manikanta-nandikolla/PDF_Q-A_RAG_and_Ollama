# Ollama
OLLAMA_BASE   = "http://localhost:11434"
LLM_MODEL     = "llama3.2:3b"
EMBED_MODEL   = "nomic-embed-text"

# Chunking
CHUNK_SIZE    = 800    # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks

# Retrieval
TOP_K         = 3      # chunks retrieved per query
CONTEXT_CHARS = 600    # max chars from each chunk sent to LLM
PROMPT_LIMIT  = 2000   # hard cap on full system prompt (chars)
HISTORY_TURNS = 1      # prior chat turns included in each request

# Embedding
EMBED_WORKERS = 4      # parallel threads for embedding (raise on fast CPUs)

# Storage
DB_PATH       = "./chroma_db"   # persistent ChromaDB directory
COLLECTION    = "pdf_rag"

# OCR
OCR_DPI       = 200    # page render resolution for scanned PDFs