"""Document conversion using the Granite Docling VLM served by llama.cpp."""

import base64
import io
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ramalama.utils.common import perror
from ramalama.utils.logger import logger

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"})
PDF_EXTENSIONS = frozenset({".pdf"})
TEXT_EXTENSIONS = frozenset({".txt", ".md", ".html", ".htm"})
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS | TEXT_EXTENSIONS


class GraniteDoclingConverter:
    """Converts documents to structured text via the Granite Docling VLM and docling-core."""

    def __init__(self, api_url: str = "http://localhost:8080"):
        self.completions_url = f"{api_url.rstrip('/')}/v1/chat/completions"

    def _send_pil_image(self, pil_image) -> str:
        """Send a PIL Image to the Granite Docling server and return raw DocTags."""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": "Convert this page to docling."},
                    ],
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.0,
        }

        req = Request(
            self.completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server returned {e.code}: {body}") from None

        return result["choices"][0]["message"]["content"]

    def convert_file(self, file_path: Path, name: str | None = None) -> list:
        """Convert a single file and return a list of documents.

        Text files (.txt, .md, .html) are read directly and returned as raw
        strings -- no VLM needed.  Images produce a single ``DoclingDocument``.
        PDFs produce one ``DoclingDocument`` per page.
        """
        suffix = file_path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return [_read_text_file(file_path)]
        if suffix in PDF_EXTENSIONS:
            return self._convert_pdf(file_path, name)
        return [self._convert_image(file_path, name)]

    def _convert_image(self, image_path: Path, name: str | None = None):
        """Convert a single image file via DocTags and return a ``DoclingDocument``."""
        DoclingDocument, DocTagsDocument = _import_docling()
        Image = _import_pillow()

        if name is None:
            name = image_path.stem

        pil_image = Image.open(image_path).convert("RGB")
        logger.debug(f"Converting {image_path.name} via Granite Docling")
        doctags = self._send_pil_image(pil_image)
        logger.debug(f"DocTags for {image_path.name} ({len(doctags)} chars)")

        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [pil_image])
        doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=name)
        logger.debug(f"DoclingDocument for {image_path.name} ({len(list(doc.iterate_items()))} items)")
        return doc

    def _convert_pdf(self, pdf_path: Path, name: str | None = None) -> list:
        """Render each page of a PDF one at a time, returning one ``DoclingDocument`` per page.

        Only a single page image is held in memory at any point, keeping RAM
        usage constant regardless of the total page count.
        """
        DoclingDocument, DocTagsDocument = _import_docling()

        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise ImportError(
                "pypdfium2 is required to process PDFs. Install with: pip install pypdfium2"
            ) from None

        if name is None:
            name = pdf_path.stem

        pdf = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(pdf)
        docs: list = []

        for page_idx in range(n_pages):
            page = pdf[page_idx]
            pil_image = page.render(scale=2).to_pil().convert("RGB")
            logger.debug(f"Converting {pdf_path.name} page {page_idx + 1}/{n_pages}")
            print(f"\r  Page {page_idx + 1}/{n_pages}...", end="", flush=True)

            doctags = self._send_pil_image(pil_image)
            logger.debug(f"DocTags for {pdf_path.name} page {page_idx + 1} ({len(doctags)} chars)")

            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [pil_image])
            doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=f"{name}_p{page_idx + 1}")
            docs.append(doc)
            del pil_image

        pdf.close()
        print()
        return docs

    def convert_directory(self, source_path: str | Path) -> list:
        """Convert every supported file under *source_path* and return a list of ``DoclingDocument`` objects."""
        source_path = Path(source_path)
        files = sorted(_collect_files(source_path))

        if not files:
            raise FileNotFoundError(f"No supported documents found in {source_path}")

        docs: list = []
        for i, fpath in enumerate(files, 1):
            perror(f"Converting {fpath.name} ({i}/{len(files)})...")
            try:
                for doc in self.convert_file(fpath):
                    if isinstance(doc, str):
                        if doc.strip():
                            docs.append(doc)
                    elif doc.export_to_markdown().strip():
                        docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to convert {fpath.name}: {e}")

        return docs


def _collect_files(path: Path) -> list[Path]:
    """Recursively collect all supported files (images and PDFs) under *path*."""
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    files: list[Path] = []
    for root, _, filenames in os.walk(path):
        for fname in filenames:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(fpath)
    return files


def _read_text_file(file_path: Path) -> str:
    """Read a text file and return its content as a string.

    HTML files have their tags stripped so only text content remains.
    """
    import re

    text = file_path.read_text(encoding="utf-8", errors="replace")
    if file_path.suffix.lower() in {".html", ".htm"}:
        text = re.sub(r"<[^>]+>", "", text)
    return text


def _import_docling():
    try:
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
    except ImportError:
        raise ImportError("docling-core is required for RAG. Install with: pip install docling-core") from None
    return DoclingDocument, DocTagsDocument


def _import_pillow():
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for RAG. Install with: pip install Pillow") from None
    return Image
