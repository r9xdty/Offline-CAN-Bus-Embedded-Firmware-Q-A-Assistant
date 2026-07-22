"""Extract plain text from uploaded documents (PDF / Markdown / plain text).

Used by the Streamlit UI so users can drop embedded-systems PDFs (datasheets, app notes,
reference-manual chapters) straight into the knowledge base. PDFs are converted to text here;
the same chunk → embed → store pipeline then applies (spec §16: convert PDFs to text first,
prefer text-heavy sections).

`pypdf` is imported lazily inside `_read_pdf_pages` so the rest of the package imports without
it, and so non-PDF uploads never pay for the dependency.
"""

from __future__ import annotations

import io
import re
from typing import List

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def _suffix(filename: str) -> str:
    return ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""


def _read_pdf_pages(data: bytes) -> List[str]:
    """Return the extracted text of each PDF page. Seam for testing (mock this)."""
    import pypdf  # lazy: heavy optional dependency, only needed for PDFs

    reader = pypdf.PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


def _clean_text(text: str) -> str:
    """Light cleanup for Markdown / plain text: keep line structure, tidy whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_pdf(text: str) -> str:
    """Reflow PDF-extracted text: de-hyphenate line wraps, join lines into flowing paragraphs.

    PDF extraction emits a newline at every visual line and often hyphenates wrapped words.
    Joining those into paragraphs (keeping blank lines as paragraph breaks) produces much
    cleaner chunks than feeding raw line-broken text to the chunker.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # join hyphenated line-wraps
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n", "\n\n", text)          # normalize paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)        # single newlines -> spaces (flow)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file's bytes, dispatched by extension.

    Raises ValueError for unsupported types. May return an empty string for image-only
    (scanned) PDFs that have no embedded text layer — callers should check and warn.
    """
    suffix = _suffix(filename)
    if suffix == ".pdf":
        return _clean_pdf("\n\n".join(_read_pdf_pages(data)))
    if suffix in {".md", ".markdown", ".txt"}:
        return _clean_text(data.decode("utf-8", errors="replace"))
    raise ValueError(
        f"Unsupported file type '{suffix or filename}'. Supported: PDF, .md, .txt"
    )
