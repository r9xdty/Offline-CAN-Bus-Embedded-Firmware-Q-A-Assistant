"""Pipeline: answer_query(question) = retrieve + generate + cite (spec §8, §4).

The `Pipeline` object loads the KB and (lazily) the Foundry models once, then reuses them for
every query so answers stay fast. `answer_query` is a module-level convenience wrapper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from . import config
from . import generate
from .retrieve import RetrievedChunk, Retriever

_CITATION = re.compile(r"\[([^\[\]]+?\.(?:md|txt))\]", re.IGNORECASE)


@dataclass
class Answer:
    """Result of a query: the grounded answer plus provenance for display/eval."""

    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.answer.strip() == config.REFUSAL_TEXT


def _unique(seq: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


class Pipeline:
    """End-to-end retrieve -> generate -> cite, holding warm resources between queries."""

    def __init__(
        self,
        db_path: Path = config.DB_PATH,
        embed_query_fn: Optional[Callable[[str], "object"]] = None,
        chat_fn: Optional[Callable[[List[dict]], str]] = None,
    ):
        self.retriever = Retriever(db_path=db_path, embed_query_fn=embed_query_fn)
        self._chat_fn = chat_fn

    @property
    def size(self) -> int:
        return self.retriever.size

    def answer(self, question: str, k: int = config.TOP_K) -> Answer:
        """Retrieve top-`k` chunks, generate a grounded answer, and attach sources."""
        question = (question or "").strip()
        if not question:
            return Answer(question=question, answer=config.REFUSAL_TEXT)

        chunks = self.retriever.retrieve(question, k=k)
        text = generate.generate_answer(question, chunks, chat_fn=self._chat_fn)

        if text.strip() == config.REFUSAL_TEXT or not chunks:
            sources: List[str] = []
        else:
            # Prefer sources the model actually cited; fall back to all retrieved sources.
            cited = _unique(re.findall(_CITATION, text))
            retrieved = _unique([c.source for c in chunks])
            sources = [s for s in retrieved if s in cited] or retrieved

        return Answer(question=question, answer=text.strip(), sources=sources, chunks=chunks)


_default_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    """Return a process-wide cached pipeline (models/KB loaded once)."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline()
    return _default_pipeline


def answer_query(question: str, k: int = config.TOP_K) -> Answer:
    """Convenience wrapper over the cached pipeline (spec §4 entry point)."""
    return get_pipeline().answer(question, k=k)
