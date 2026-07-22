"""Retrieval: embed the query -> cosine over stored vectors -> top-K chunks (spec §8.2).

For this corpus size (dozens-to-low-hundreds of chunks) a brute-force NumPy scan is correct
and fast; no ANN index or external vector DB (out of scope, spec §7).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from . import config
from . import vectors


@dataclass
class RetrievedChunk:
    """A retrieved chunk with provenance and similarity score."""

    id: int
    source: str
    chunk_index: int
    content: str
    score: float


def load_matrix(db_path: Path = config.DB_PATH):
    """Load all chunks and stack their (already normalized) embeddings into one matrix.

    Returns (rows, matrix) where `rows` is a list of (id, source, chunk_index, content) and
    `matrix` is a float32 array of shape (n, dim). Empty KB -> ([], empty array).
    """
    conn = sqlite3.connect(db_path)
    try:
        # A fresh checkout (before `python -m src.ingest` runs) has no table yet — treat
        # that as an empty KB rather than crashing, so the CLI/UI can prompt to ingest.
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if not exists:
            return [], np.empty((0, 0), dtype=np.float32)
        cur = conn.execute(
            "SELECT id, source, chunk_index, content, embedding FROM chunks ORDER BY id"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)

    meta = [(r[0], r[1], r[2], r[3]) for r in rows]
    matrix = np.vstack([vectors.from_blob(r[4]) for r in rows]).astype(np.float32)
    return meta, matrix


def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> List[tuple[int, float]]:
    """Return the (row_index, score) of the top-`k` rows by cosine similarity.

    `query_vec` is normalized here; `matrix` rows are assumed pre-normalized, so cosine
    similarity reduces to a dot product. Results are sorted by descending score.
    """
    if matrix.size == 0 or k <= 0:
        return []
    q = vectors.l2_normalize(query_vec)
    scores = matrix @ q
    k = min(k, scores.shape[0])
    # argpartition for the top-k, then sort just those k by score descending.
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx]


def _default_embed_query(text: str) -> np.ndarray:
    from . import foundry_client

    return foundry_client.embed_query(text)


class Retriever:
    """Loads the KB once and answers top-K similarity queries against it."""

    def __init__(
        self,
        db_path: Path = config.DB_PATH,
        embed_query_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.db_path = db_path
        self._embed_query = embed_query_fn or _default_embed_query
        self._meta, self._matrix = load_matrix(db_path)

    @property
    def size(self) -> int:
        return len(self._meta)

    def retrieve(self, question: str, k: int = config.TOP_K) -> List[RetrievedChunk]:
        """Return the top-`k` most relevant chunks for `question`."""
        if not question or not question.strip():
            return []
        if self.size == 0:
            return []
        query_vec = np.asarray(self._embed_query(question), dtype=np.float32)
        hits = cosine_top_k(query_vec, self._matrix, k)
        results: List[RetrievedChunk] = []
        for row_idx, score in hits:
            cid, source, chunk_index, content = self._meta[row_idx]
            results.append(
                RetrievedChunk(
                    id=cid,
                    source=source,
                    chunk_index=chunk_index,
                    content=content,
                    score=score,
                )
            )
        return results
