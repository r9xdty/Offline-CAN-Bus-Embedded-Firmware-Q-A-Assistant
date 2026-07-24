"""Pipeline: answer_query(question) = retrieve + generate + cite (spec §8, §4).

The `Pipeline` object loads the KB and (lazily) the Foundry models once, then reuses them for
every query so answers stay fast. `answer_query` is a module-level convenience wrapper.

Retrieved chunks below `config.MIN_SCORE` are dropped before generation: an off-topic question
then reaches the model with no context (or none at all), yielding a clean, deterministic
refusal instead of relying solely on the LLM to notice weak, semi-relevant chunks.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from . import config
from . import generate
from .retrieve import RetrievedChunk, Retriever

# Citations look like [source_name.ext]; match the extensions we actually ingest.
_CITATION = re.compile(r"\[([^\[\]]+?\.(?:md|markdown|txt|pdf))\]", re.IGNORECASE)

# A prior conversation turn: (user_question, assistant_answer).
Turn = Tuple[str, str]

# Opening words that signal an elliptical follow-up ("explain that…", "and the extended one?",
# "it uses which protocol?") whose retrieval benefits from the previous question's context.
# Only the FIRST word is checked, so a self-contained question that merely contains "it"/"that"
# ("…how is it structured?") is left untouched.
_FOLLOWUP_STARTS = {
    "and", "also", "or", "but", "then", "so",
    "it", "its", "that", "this", "those", "these", "them", "they", "he", "she",
    "explain", "elaborate", "expand", "clarify",
}


@dataclass
class Answer:
    """Result of a query: the grounded answer plus provenance for display/eval."""

    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)
    elapsed_s: float = 0.0
    mode: str = ""
    kind: str = "grounded"  # "grounded" | "general" | "refusal"

    @property
    def is_refusal(self) -> bool:
        return self.answer.strip() == config.REFUSAL_TEXT

    @property
    def is_general(self) -> bool:
        return self.kind == "general"

    @property
    def top_score(self) -> Optional[float]:
        """Highest similarity among the retrieved chunks (None if nothing retrieved)."""
        return max((c.score for c in self.chunks), default=None)


def _retrieval_query(question: str, history: Optional[Sequence[Turn]]) -> str:
    """Expand an elliptical follow-up with the previous question so retrieval still lands.

    A self-contained question is used as-is; a short or pronoun-y follow-up ("what about at
    250 kbps?", "explain that") is prefixed with the last user question so the embedding
    carries the topic.
    """
    if not history:
        return question
    words = [w.strip("?.,!:;").lower() for w in question.split()]
    elliptical = len(words) < 6 or (bool(words) and words[0] in _FOLLOWUP_STARTS)
    if not elliptical:
        return question
    prev_question = next((q for q, _ in reversed(list(history)) if q and q.strip()), "")
    return f"{prev_question} {question}".strip() if prev_question else question


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

    def answer(
        self,
        question: str,
        history: Optional[Sequence[Turn]] = None,
        mode: str | None = None,
        k: int = config.TOP_K,
        on_token: Optional[Callable[[str], None]] = None,
        general_enabled: Optional[bool] = None,
    ) -> Answer:
        """Retrieve top-`k` chunks, drop weak matches, generate a grounded answer, cite.

        `history` is prior (question, answer) turns for follow-up continuity; `mode` is
        "short" or "explain" (falls back to the configured default). `on_token`, if given,
        streams each answer delta as it arrives. `general_enabled`, if given, overrides
        `config.GENERAL_KNOWLEDGE_ENABLED` for this call only (lets the CLI/Streamlit toggle
        the general-knowledge tier live without touching the env-driven master switch).
        """
        start = time.perf_counter()
        mode = mode or config.DEFAULT_MODE
        question = (question or "").strip()
        if not question:
            if on_token is not None:
                on_token(config.REFUSAL_TEXT)
            return Answer(
                question=question,
                answer=config.REFUSAL_TEXT,
                elapsed_s=time.perf_counter() - start,
                mode=mode,
                kind="refusal",
            )

        retrieved = self.retriever.retrieve(_retrieval_query(question, history), k=k)
        # Keep only chunks that clear the similarity floor; feed just those to the model.
        relevant = [c for c in retrieved if c.score >= config.MIN_SCORE]
        # The general-knowledge tier only opens when the question is relevant enough to have
        # SOME context to attempt first, AND the top match clears the (stricter) domain floor --
        # i.e. the question reads as clearly on-topic, not merely a weak keyword overlap.
        top = max((c.score for c in retrieved), default=None)
        enabled = config.GENERAL_KNOWLEDGE_ENABLED if general_enabled is None else general_enabled
        allow_general = (
            bool(relevant)
            and enabled
            and top is not None
            and top >= config.DOMAIN_SCORE
        )
        text = generate.generate_answer(
            question, relevant, history=history, mode=mode,
            chat_fn=self._chat_fn, on_token=on_token, allow_general=allow_general,
        ).strip()

        if text == config.REFUSAL_TEXT:
            kind = "refusal"
            sources: List[str] = []
        elif config.is_general_answer(text):
            kind = "general"
            sources = []
        elif not relevant:
            # Defensive: shouldn't normally happen (no relevant chunks -> generate_answer
            # refuses before calling the model), but keep classification airtight either way.
            kind = "refusal"
            sources = []
        else:
            kind = "grounded"
            # Prefer sources the model actually cited; fall back to the chunks we used.
            cited = _unique(re.findall(_CITATION, text))
            used = _unique([c.source for c in relevant])
            sources = [s for s in used if s in cited] or used

        return Answer(
            question=question,
            answer=text,
            sources=sources,
            chunks=retrieved,  # show every retrieved chunk (with scores) for transparency
            elapsed_s=time.perf_counter() - start,
            mode=mode,
            kind=kind,
        )


_default_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    """Return a process-wide cached pipeline (models/KB loaded once)."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline()
    return _default_pipeline


def answer_query(
    question: str,
    history: Optional[Sequence[Turn]] = None,
    mode: str | None = None,
    k: int = config.TOP_K,
    on_token: Optional[Callable[[str], None]] = None,
    general_enabled: Optional[bool] = None,
) -> Answer:
    """Convenience wrapper over the cached pipeline (spec §4 entry point)."""
    return get_pipeline().answer(
        question, history=history, mode=mode, k=k, on_token=on_token,
        general_enabled=general_enabled,
    )
