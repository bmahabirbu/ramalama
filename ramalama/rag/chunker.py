"""Section-aware document chunking.

Handles both ``DoclingDocument`` objects (from PDFs/images) and raw text
strings (from .txt, .md, .html files).  In both cases content is grouped
by heading so each chunk is a coherent section.
"""

import hashlib
import re
import uuid

from docling_core.types.doc.document import SectionHeaderItem, TitleItem

_HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)


def chunk_documents(docs: list) -> tuple[list[str], list[int]]:
    """Chunk a mixed list of ``DoclingDocument`` objects and raw text strings.

    Returns ``(chunks, ids)`` where *ids* are deterministic ``int64`` hashes.
    """
    chunks: list[str] = []
    ids: list[int] = []

    for doc in docs:
        if isinstance(doc, str):
            _chunk_text(doc, chunks, ids)
        else:
            _chunk_docling_doc(doc, chunks, ids)

    return chunks, ids


def _chunk_docling_doc(doc, chunks: list[str], ids: list[int]):
    """Section-aware chunking via the DoclingDocument layout tree."""
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


def _chunk_text(text: str, chunks: list[str], ids: list[int]):
    """Section-aware chunking for raw markdown/text strings.

    Splits on markdown headings (``# ...``).  Plain text without headings
    is split on blank-line paragraph boundaries.
    """
    if _HEADING_RE.search(text):
        sections = _HEADING_RE.split(text)
        buf = sections[0].strip()
        if buf:
            chunks.append(buf)
            ids.append(_text_hash(buf))

        i = 1
        while i < len(sections) - 1:
            hashes = sections[i]
            body = sections[i + 1].strip()
            if body:
                section = f"{hashes} {body}"
                chunks.append(section)
                ids.append(_text_hash(section))
            i += 2
    else:
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                chunks.append(para)
                ids.append(_text_hash(para))


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
