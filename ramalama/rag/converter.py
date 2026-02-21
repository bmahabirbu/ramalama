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
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


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

    def convert_file(self, file_path: Path, name: str | None = None) -> str:
        """Convert a single file (PDF or image) and return markdown text."""
        if file_path.suffix.lower() in PDF_EXTENSIONS:
            return self._convert_pdf(file_path, name)
        return self._convert_image(file_path, name)

    def _convert_image(self, image_path: Path, name: str | None = None) -> str:
        """Convert a single image file via DocTags and return markdown."""
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
        md = doc.export_to_markdown()
        logger.debug(f"Markdown for {image_path.name} ({len(md)} chars)")
        return md

    def _convert_pdf(self, pdf_path: Path, name: str | None = None) -> str:
        """Render each page of a PDF to an image, send to VLM, return markdown."""
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

        all_doctags: list[str] = []
        all_images: list = []

        for page_idx in range(n_pages):
            page = pdf[page_idx]
            pil_image = page.render(scale=2).to_pil().convert("RGB")
            logger.debug(f"Converting {pdf_path.name} page {page_idx + 1}/{n_pages}")
            print(f"  Page {page_idx + 1}/{n_pages}...", end=" ", flush=True)

            doctags = self._send_pil_image(pil_image)
            logger.debug(f"DocTags for {pdf_path.name} page {page_idx + 1} ({len(doctags)} chars)")
            all_doctags.append(doctags)
            all_images.append(pil_image)

        pdf.close()

        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(all_doctags, all_images)
        doc = DoclingDocument.load_from_doctags(doctags_doc, document_name=name)
        md = doc.export_to_markdown()
        logger.debug(f"Markdown for {pdf_path.name} ({len(md)} chars)")
        return md

    def convert_directory(self, source_path: str | Path) -> list[str]:
        """Convert every supported file under *source_path* and return a list of markdown strings."""
        source_path = Path(source_path)
        files = sorted(_collect_files(source_path))

        if not files:
            raise FileNotFoundError(f"No supported documents found in {source_path}")

        texts: list[str] = []
        for i, fpath in enumerate(files, 1):
            perror(f"Converting {fpath.name} ({i}/{len(files)})...")
            try:
                text = self.convert_file(fpath)
                if text.strip():
                    texts.append(text)
                else:
                    logger.warning(f"{fpath.name} produced no text")
            except Exception as e:
                logger.warning(f"Failed to convert {fpath.name}: {e}")

        return texts


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
