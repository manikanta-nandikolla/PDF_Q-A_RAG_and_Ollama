import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import OLLAMA_BASE, EMBED_MODEL, LLM_MODEL, EMBED_WORKERS


def ollama_embed(text: str) -> list[float]:
    """Embed a single text string via Ollama."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def ollama_embed_batch(chunks: list[str]) -> list[list[float]]:
    """
    Embed multiple chunks in parallel using a thread pool.
    Order of returned vectors matches order of input chunks.
    """
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {pool.submit(ollama_embed, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def ollama_chat(messages: list[dict]) -> str:
    """
    Send a list of role/content message dicts to the Ollama chat API.
    Returns the assistant's reply as a plain string.
    """
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]