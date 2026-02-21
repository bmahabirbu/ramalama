"""Section-aware document chunking via DoclingDocument's layout tree.

Groups content under its heading so each chunk is a coherent section
(heading + paragraphs + tables + captions).  No extra dependencies
beyond docling-core.
"""

import hashlib
import uuid

from docling_core.types.doc.document import SectionHeaderItem, TitleItem


def chunk_documents(docs: list) -> tuple[list[str], list[int]]:
    """Chunk ``DoclingDocument`` objects by section.

    Content is accumulated under each heading and flushed as a single
    chunk when the next heading (at the same or higher level) appears.

    Returns ``(chunks, ids)`` where *ids* are deterministic ``int64`` hashes.
    """
    chunks: list[str] = []
    ids: list[int] = []

    for doc in docs:
        parts: list[str] = []

        for item, _level in doc.iterate_items():
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                _flush(parts, chunks, ids)
                parts = [item.text.strip()] if item.text else []
                continue

            if hasattr(item, "text") and item.text:
                text = item.text.strip()
                if text:
                    parts.append(text)

        _flush(parts, chunks, ids)

    return chunks, ids


def _flush(parts: list[str], chunks: list[str], ids: list[int]):
    """Join accumulated section parts into a chunk and append it."""
    combined = "\n".join(parts).strip()
    if combined:
        chunks.append(combined)
        ids.append(_text_hash(combined))
    parts.clear()


def _text_hash(text: str) -> int:
    """Deterministic signed-int64 hash derived from the SHA-256 of *text*."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return uuid.UUID(digest[:32]).int & ((1 << 63) - 1)
