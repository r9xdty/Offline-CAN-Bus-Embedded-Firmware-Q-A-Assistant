"""Ingestion: read docs from data/raw/ -> chunk -> embed -> write to SQLite (spec §8.1).

Run as a module::

    python -m src.ingest            # rebuild data/kb.sqlite from data/raw/
    python -m src.ingest --debug    # also print a sample of chunks

The chunker (`chunk_text`) and DB writer (`write_chunks`) are pure and take their inputs
directly, so ingestion logic can be unit-tested without the Foundry runtime by passing a
fake `embed_fn`.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from . import config
from . import vectors

# A "chunk record" prior to embedding: (source, chunk_index, content).
ChunkRecord = Tuple[str, int, str]

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")


# --------------------------------------------------------------------------- #
# Chunking (pure)
# --------------------------------------------------------------------------- #
def _split_blocks(text: str) -> List[str]:
    """Split into paragraph blocks on blank lines, attaching a heading to the block below."""
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks: List[str] = []
    pending_heading = ""
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        # A lone heading line glues onto the following block so its context travels with it.
        if len(lines) == 1 and _HEADING.match(lines[0]):
            pending_heading = lines[0].strip()
            continue
        if pending_heading:
            block = f"{pending_heading}\n{block}"
            pending_heading = ""
        blocks.append(block)
    if pending_heading:
        blocks.append(pending_heading)
    return blocks


def _split_sentences(block: str) -> List[str]:
    """Split an over-long block into sentence-ish pieces so we avoid cutting mid-sentence."""
    parts = [p.strip() for p in _SENTENCE_END.split(block) if p.strip()]
    return parts or [block.strip()]


def _overlap_tail(text: str, overlap: int) -> str:
    """Return roughly the last `overlap` chars of `text`, trimmed to a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if len(text) <= overlap else text[-overlap:]
    tail = text[-overlap:]
    # Advance to the next word boundary so we don't start mid-word.
    space = tail.find(" ")
    if space != -1:
        tail = tail[space + 1 :]
    return tail.strip()


def chunk_text(
    text: str,
    target: int = config.CHUNK_TARGET_CHARS,
    max_chars: int = config.CHUNK_MAX_CHARS,
    overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """Split `text` into ~`target`-char chunks (<= `max_chars`) with ~`overlap` char overlap.

    Splits on headings/paragraphs, then sentences for over-long paragraphs. Whitespace-only
    fragments are skipped. Consecutive chunks share a trailing-sentence overlap so a fact
    that straddles a boundary still lands whole in at least one chunk.
    """
    units: List[str] = []
    for block in _split_blocks(text):
        if len(block) <= max_chars:
            units.append(block)
        else:
            units.extend(_split_sentences(block))

    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            carry = _overlap_tail(current, overlap)
            current = f"{carry}\n\n{unit}".strip() if carry else unit
        else:
            current = candidate
        # Emit early once we're comfortably past the target to keep chunks tight.
        if len(current) >= target and len(current) >= max_chars - overlap:
            chunks.append(current.strip())
            current = _overlap_tail(current, overlap)
    if current.strip():
        chunks.append(current.strip())

    # Drop any empty / whitespace-only chunks and de-dupe adjacent identical carries.
    cleaned: List[str] = []
    for ch in chunks:
        ch = ch.strip()
        if ch and (not cleaned or cleaned[-1] != ch):
            cleaned.append(ch)
    return cleaned


def build_records(raw_dir: Path) -> List[ChunkRecord]:
    """Read every .md/.txt file in `raw_dir` and return ordered chunk records."""
    records: List[ChunkRecord] = []
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in {".md", ".txt"})
    for path in files:
        text = path.read_text(encoding="utf-8")
        for idx, content in enumerate(chunk_text(text)):
            records.append((path.name, idx, content))
    return records


# --------------------------------------------------------------------------- #
# SQLite (pure given an embeddings matrix)
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    source      TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT    NOT NULL,
    embedding   BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source ON chunks(source);
"""


def _rows_for(records: Sequence[ChunkRecord], embeddings: np.ndarray):
    """Build INSERT rows, L2-normalizing each embedding so similarity is a dot product."""
    return [
        (source, chunk_index, content, vectors.to_blob(vectors.l2_normalize(embeddings[i])))
        for i, (source, chunk_index, content) in enumerate(records)
    ]


def write_chunks(
    db_path: Path,
    records: Sequence[ChunkRecord],
    embeddings: np.ndarray,
) -> int:
    """(Re)build the SQLite KB from records + their embeddings. Idempotent.

    Clears the whole table and repopulates it. Returns the number of chunks written.
    """
    if len(records) != len(embeddings):
        raise ValueError(
            f"records ({len(records)}) and embeddings ({len(embeddings)}) length mismatch"
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        # Idempotent rebuild: clear and repopulate in one transaction.
        conn.execute("DELETE FROM chunks")
        conn.executemany(
            "INSERT INTO chunks (source, chunk_index, content, embedding) VALUES (?, ?, ?, ?)",
            _rows_for(records, embeddings),
        )
        conn.commit()
    finally:
        conn.close()
    return len(records)


def add_document(
    source: str,
    text: str,
    db_path: Path = config.DB_PATH,
    embed_fn: Callable[[List[str]], np.ndarray] | None = None,
) -> int:
    """Chunk, embed, and upsert one document into the KB **without wiping the rest**.

    Rows for an existing `source` are replaced (idempotent re-upload). Used by the Streamlit
    uploader to add PDFs/notes on top of the existing corpus. Returns the chunk count added.
    """
    embed = embed_fn or _default_embed
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = np.asarray(embed(chunks), dtype=np.float32)
    records: List[ChunkRecord] = [(source, i, content) for i, content in enumerate(chunks)]

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        conn.executemany(
            "INSERT INTO chunks (source, chunk_index, content, embedding) VALUES (?, ?, ?, ?)",
            _rows_for(records, embeddings),
        )
        conn.commit()
    finally:
        conn.close()
    return len(records)


def list_sources(db_path: Path = config.DB_PATH) -> List[Tuple[str, int]]:
    """Return [(source, chunk_count)] for everything in the KB (empty if not built yet)."""
    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if not exists:
            return []
        return conn.execute(
            "SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source"
        ).fetchall()
    finally:
        conn.close()


def remove_source(source: str, db_path: Path = config.DB_PATH) -> int:
    """Delete all chunks for `source`. Returns the number of rows removed."""
    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if not exists:
            return 0
        cur = conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _default_embed(texts: List[str]) -> np.ndarray:
    from . import foundry_client

    return foundry_client.embed_texts(texts)


def ingest(
    raw_dir: Path = config.RAW_DIR,
    db_path: Path = config.DB_PATH,
    embed_fn: Callable[[List[str]], np.ndarray] | None = None,
) -> int:
    """Full ingestion pass: read -> chunk -> embed -> store. Returns the chunk count.

    `embed_fn` defaults to the Foundry NVIDIA embedding model; pass a fake in tests.
    """
    embed = embed_fn or _default_embed
    records = build_records(raw_dir)
    if not records:
        raise RuntimeError(f"No .md/.txt documents found in {raw_dir}")
    contents = [content for (_, _, content) in records]
    embeddings = np.asarray(embed(contents), dtype=np.float32)
    return write_chunks(db_path, records, embeddings)


def _sample(records_preview: Iterable[Tuple[str, int, str]], n: int = 3) -> None:
    for source, idx, content in list(records_preview)[:n]:
        preview = content.replace("\n", " ")
        print(f"  [{source}#{idx}] {preview[:120]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CAN-bus RAG knowledge base.")
    parser.add_argument("--debug", action="store_true", help="print a sample of chunks")
    args = parser.parse_args()

    if args.debug:
        preview = build_records(config.RAW_DIR)
        print(f"Parsed {len(preview)} chunks from {config.RAW_DIR}. Sample:")
        _sample(preview)

    count = ingest()
    print(f"Ingested {count} chunks into {config.DB_PATH}")


if __name__ == "__main__":
    main()
