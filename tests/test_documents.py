"""Offline tests for document extraction + incremental knowledge-base upload.

The PDF path is exercised by mocking `documents._read_pdf_pages`, so these run without pypdf
or the Foundry runtime. The incremental ingest (`add_document` / `list_sources` /
`remove_source`) uses a real temp SQLite DB with a deterministic fake embedder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src import documents, ingest


# Reuse the bag-of-words fake embedder shape from the pipeline tests.
VOCAB = ["length", "identifier", "error", "timing", "canopen", "j1939", "transceiver", "fd"]


def _bow(text: str) -> np.ndarray:
    low = text.lower()
    vec = np.array([low.count(w) for w in VOCAB], dtype=np.float32)
    return vec + 1e-3 if not vec.any() else vec


def bow_embed(texts: List[str]) -> np.ndarray:
    return np.vstack([_bow(t) for t in texts]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_txt_and_md():
    txt = documents.extract_text("note.txt", b"CAN termination is 120 ohm.\n\n\n\nNext.")
    assert "120 ohm" in txt
    assert "\n\n\n" not in txt  # collapsed blank runs

    md = documents.extract_text("note.md", b"# Title\n\nSome content about bit timing.")
    assert md.startswith("# Title")  # markdown structure preserved


def test_extract_unsupported_type_raises():
    with pytest.raises(ValueError):
        documents.extract_text("firmware.bin", b"\x00\x01\x02")


def test_extract_pdf_dehyphenates_and_flows(monkeypatch):
    # Simulate pypdf returning line-broken, hyphenated page text.
    pages = [
        "STM32 bit timing config-\nuration uses BRP,\nTSEG1 and TSEG2.",
        "\n\nThe sample point is\nabout 87.5%.",
    ]
    monkeypatch.setattr(documents, "_read_pdf_pages", lambda data: pages)

    text = documents.extract_text("stm32.pdf", b"%PDF-fake-bytes")
    assert "configuration uses BRP," in text          # hyphenation join + line flow
    assert "TSEG1 and TSEG2." in text
    assert "sample point is about 87.5%." in text
    assert "-\n" not in text


def test_extract_pdf_empty_when_no_text_layer(monkeypatch):
    monkeypatch.setattr(documents, "_read_pdf_pages", lambda data: ["", "   ", "\n"])
    assert documents.extract_text("scan.pdf", b"%PDF").strip() == ""


# --------------------------------------------------------------------------- #
# Incremental ingest (upsert / list / remove)
# --------------------------------------------------------------------------- #
def test_add_document_upserts_without_wiping(tmp_path: Path):
    db = tmp_path / "kb.sqlite"

    n1 = ingest.add_document("a.pdf", "error error error confinement counters", db, bow_embed)
    n2 = ingest.add_document("b.pdf", "timing timing timing segments quanta", db, bow_embed)
    assert n1 >= 1 and n2 >= 1

    sources = dict(ingest.list_sources(db))
    assert set(sources) == {"a.pdf", "b.pdf"}  # both present, neither wiped the other

    # Re-uploading the same source replaces its rows (idempotent), doesn't duplicate.
    ingest.add_document("a.pdf", "error error error confinement counters", db, bow_embed)
    assert dict(ingest.list_sources(db))["a.pdf"] == n1


def test_add_document_empty_text_adds_nothing(tmp_path: Path):
    db = tmp_path / "kb.sqlite"
    assert ingest.add_document("empty.pdf", "   \n  ", db, bow_embed) == 0
    assert ingest.list_sources(db) == []


def test_remove_source(tmp_path: Path):
    db = tmp_path / "kb.sqlite"
    ingest.add_document("a.pdf", "error error error", db, bow_embed)
    ingest.add_document("b.pdf", "timing timing timing", db, bow_embed)

    removed = ingest.remove_source("a.pdf", db)
    assert removed >= 1
    assert set(dict(ingest.list_sources(db))) == {"b.pdf"}


def test_list_and_remove_on_missing_db_are_safe(tmp_path: Path):
    db = tmp_path / "never.sqlite"
    assert ingest.list_sources(db) == []
    assert ingest.remove_source("x.pdf", db) == 0


def test_uploaded_document_is_retrievable(tmp_path: Path):
    """End-to-end (minus Foundry): upsert a doc, then retrieve it by keyword."""
    from src.retrieve import Retriever

    db = tmp_path / "kb.sqlite"
    ingest.add_document("timing_notes.pdf", "bit timing segments and quanta timing timing", db, bow_embed)
    ingest.add_document("error_notes.pdf", "error confinement counters error error", db, bow_embed)

    retriever = Retriever(db_path=db, embed_query_fn=lambda q: _bow(q))
    hits = retriever.retrieve("tell me about timing", k=1)
    assert hits[0].source == "timing_notes.pdf"
