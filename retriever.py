# ── retriever.py ──────────────────────────────────────────────────────────────
# Handles everything after a question is asked:
#   - answer_question()   embed query → retrieve chunks → build prompt → LLM
#   - sources_md()        format retrieved chunks + relevance scores for UI
#   - export_chat()       save chat history to a .md file for download

import gradio as gr
from datetime import datetime

from config import TOP_K, CONTEXT_CHARS, PROMPT_LIMIT, HISTORY_TURNS
from embedder import ollama_embed, ollama_chat
from ingestor import collection


def answer_question(question: str, history: list):
    """
    Gradio event handler: answer a question using RAG.
    Returns (history, cleared_input, interactive_update, sources_markdown).
    """
    if not question.strip():
        return history, "", gr.update(), sources_md([], [], [])

    if collection.count() == 0:
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": "⚠️ No PDFs ingested yet. Please upload and ingest files first."})
        return history, "", gr.update(), sources_md([], [], [])

    # Embed the question and retrieve top-k chunks with distances
    q_emb   = ollama_embed(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # Build context block for the system prompt
    context_parts = [
        f"[Source: {meta.get('source', '?')}]\n{doc[:CONTEXT_CHARS]}"
        for doc, meta in zip(docs, metas)
    ]
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "Answer using only the context below. Cite the source filename. "
        "If the answer is absent, say so.\n\n"
        f"CONTEXT:\n{context}"
    )[:PROMPT_LIMIT]

    # Build messages list: system + last N turns + new question
    messages = [{"role": "system", "content": system_prompt}]

    prior = [
        m for m in history[-(HISTORY_TURNS * 2):]
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]
    for msg in prior:
        messages.append({"role": msg["role"], "content": msg["content"][:400]})

    messages.append({"role": "user", "content": question})

    answer = ollama_chat(messages)
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    return history, "", gr.update(interactive=True), sources_md(docs, metas, distances)


def sources_md(docs: list, metas: list, distances: list = None) -> str:
    """
    Format retrieved chunks as a readable markdown panel with relevance scores.
    L2 distance is converted to a 0–100% similarity score.
    """
    if not docs:
        return "*Ask a question to see retrieved source chunks here.*"

    lines = ["**Retrieved source chunks:**\n"]
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        source = meta.get("source", "?")
        score  = ""
        if distances:
            sim   = max(0, round((1 - distances[i] / 2) * 100, 1))
            score = f" · **{sim}% match**"
        lines.append(f"**[{i + 1}] {source}**{score}")
        lines.append(f"> {doc[:300].strip()}…\n")

    return "\n".join(lines)


def export_chat(history: list) -> str:
    """
    Write the full chat history to chat_export.md and return the file path
    so Gradio's DownloadButton can serve it.
    """
    if not history:
        return None

    lines = [f"# PDF Q&A Chat Export\n*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"]
    for msg in history:
        role = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{role}\n{msg['content']}\n")

    out_path = "chat_export.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(lines))

    return out_path