import os
import gradio as gr
from pathlib import Path

import chromadb

from config import CHUNK_SIZE, CHUNK_OVERLAP, DB_PATH, COLLECTION
from ocr import extract_pdf_text, OCR_AVAILABLE
from embedder import ollama_embed_batch

# ── Persistent ChromaDB setup ─────────────────────────────────────────────────
os.makedirs(DB_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=DB_PATH)


def _load_collection():
    try:
        return chroma_client.get_collection(COLLECTION)
    except Exception:
        return chroma_client.create_collection(COLLECTION)


def _reset_collection():
    try:
        chroma_client.delete_collection(COLLECTION)
    except Exception:
        pass
    return chroma_client.create_collection(COLLECTION)


# Module-level shared state — imported by retriever.py and app.py
collection = _load_collection()


def _load_ingested_files() -> list[str]:
    """Restore ingested file list from ChromaDB metadata on startup."""
    try:
        meta = collection.get(limit=9999, include=["metadatas"])
        return list({m["source"] for m in meta["metadatas"] if m.get("source")})
    except Exception:
        return []


ingested_files: list[str] = _load_ingested_files()


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of fixed character size."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 60]


# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest_pdfs(pdf_files, progress=gr.Progress()):
    """
    Gradio event handler: ingest one or more PDF files.
    - Skips files already in the store.
    - Extracts text (with OCR fallback for scanned pages).
    - Embeds chunks in parallel and stores them in ChromaDB.
    Returns (status_markdown, msg_box_update, file_list_markdown).
    """
    global collection, ingested_files

    if not pdf_files:
        return "No files uploaded.", gr.update(interactive=False), file_list_md()

    new_names = [Path(f.name).name for f in pdf_files]
    already   = [n for n in new_names if n in ingested_files]
    fresh     = [f for f in pdf_files if Path(f.name).name not in ingested_files]

    if not fresh:
        return (
            f"⚠️ Already ingested: {', '.join(already)}",
            gr.update(interactive=True),
            file_list_md(),
        )

    log_lines = []

    for fi, pdf_file in enumerate(fresh):
        fname = Path(pdf_file.name).name
        progress(fi / len(fresh), desc=f"Reading {fname}…")

        text, ocr_used = extract_pdf_text(pdf_file.name)
        if not text.strip():
            note = (
                " (skipped — install Tesseract for scanned PDFs)"
                if not OCR_AVAILABLE
                else " (skipped — no extractable text)"
            )
            log_lines.append(f"⚠️ **{fname}**{note}")
            continue

        chunks   = chunk_text(text)
        ocr_note = " *(OCR)*" if ocr_used else ""
        progress(
            (fi + 0.1) / len(fresh),
            desc=f"Embedding {len(chunks)} chunks in parallel… ({fname})",
        )

        embeddings  = ollama_embed_batch(chunks)
        progress((fi + 0.9) / len(fresh), desc=f"Storing {fname} in ChromaDB…")

        safe_name   = fname.replace(" ", "_").replace(".", "_")
        base_offset = collection.count()
        ids         = [f"{safe_name}_{base_offset + i}" for i in range(len(chunks))]
        metadatas   = [{"source": fname}] * len(chunks)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        ingested_files.append(fname)
        log_lines.append(f"✅ **{fname}** — {len(chunks)} chunks{ocr_note}")

    if already:
        log_lines.append(f"ℹ️ Skipped (already loaded): {', '.join(already)}")

    summary = (
        "\n".join(log_lines)
        + f"\n\n**Total chunks: {collection.count()}** *(saved to disk)*"
    )
    return summary, gr.update(interactive=True), file_list_md()


def clear_all():
    """
    Wipe the ChromaDB collection and reset ingested file list.
    Returns values for: chatbot, msg_box, status_box, msg_box, file_list.
    """
    global collection, ingested_files
    collection     = _reset_collection()
    ingested_files = []
    return (
        [],
        "",
        "🗑️ Cleared. Upload new PDFs to start again.",
        gr.update(interactive=False),
        file_list_md(),
    )


def file_list_md() -> str:
    """Return a markdown string listing all currently ingested PDFs."""
    if not ingested_files:
        return "*No PDFs loaded yet.*"
    return "**Loaded PDFs:**\n" + "\n".join(f"- {n}" for n in ingested_files)