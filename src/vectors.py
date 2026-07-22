"""Small vector helpers shared by ingestion and retrieval.

Embeddings are stored L2-normalized as raw float32 bytes (spec §7). Normalizing once at
ingestion turns query-time cosine similarity into a plain dot product.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return a float32 copy of `vec` scaled to unit L2 norm.

    A zero vector is returned unchanged (as float32) to avoid divide-by-zero.
    """
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def to_blob(vec: np.ndarray) -> bytes:
    """Serialize an (already normalized) vector to raw float32 bytes for SQLite."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Deserialize raw float32 bytes back into a 1-D NumPy vector."""
    return np.frombuffer(blob, dtype=np.float32)
