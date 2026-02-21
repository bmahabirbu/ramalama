"""Text chunking for the RAG pipeline."""

import hashlib
import uuid


def chunk_texts(texts: list[str], max_tokens: int = 128, overlap: int = 32) -> tuple[list[str], list[int]]:
    """Split a list of document texts into overlapping chunks.

    Returns ``(chunks, ids)`` where *ids* are deterministic ``int64`` hashes.
    """
    chunks: list[str] = []
    ids: list[int] = []

    for text in texts:
        for chunk in _split_text(text, max_tokens=max_tokens, overlap=overlap):
            chunks.append(chunk)
            ids.append(_text_hash(chunk))

    return chunks, ids


def _split_text(text: str, max_tokens: int = 128, overlap: int = 32) -> list[str]:
    """Split *text* into overlapping chunks using whitespace-based tokenisation."""
    words = text.split()
    if not words:
        return []

    result: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            result.append(chunk)
        if end >= len(words):
            break
        start += max_tokens - overlap

    return result


def _text_hash(text: str) -> int:
    """Deterministic signed-int64 hash derived from the SHA-256 of *text*."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return uuid.UUID(digest[:32]).int & ((1 << 63) - 1)
