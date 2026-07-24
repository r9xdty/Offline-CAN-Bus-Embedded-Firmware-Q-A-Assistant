"""Generation: build a grounded prompt from retrieved chunks -> call chat -> answer (spec §8.3).

Supports conversation memory (prior turns feed reference resolution for follow-ups) and two
answer modes ("short" vs "explain") that shape style + length without touching the grounded /
refusal contract. Pure prompt assembly is separated from the model call so it can be
unit-tested with a fake chat function.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from . import config
from .retrieve import RetrievedChunk

# A prior conversation turn: (user_question, assistant_answer).
Turn = Tuple[str, str]


def fit_context(
    chunks: Sequence[RetrievedChunk],
    char_budget: int | None = None,
) -> List[RetrievedChunk]:
    """Keep the highest-ranked chunks whose labeled text fits the context char budget.

    Chunks arrive already sorted by score. We add them in order and stop before exceeding the
    budget, guaranteeing the whole prompt stays inside the 4K token window (spec §8.2).
    """
    budget = config.context_char_budget() if char_budget is None else char_budget
    kept: List[RetrievedChunk] = []
    used = 0
    for ch in chunks:
        block_len = len(_format_chunk(ch)) + 2  # +2 for the joining blank line
        if kept and used + block_len > budget:
            break
        kept.append(ch)
        used += block_len
    return kept


def _format_chunk(chunk: RetrievedChunk) -> str:
    """Prefix a chunk with its source label so the model can cite it (spec §9)."""
    return f"[{chunk.source}]\n{chunk.content}"


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Join labeled chunks into the CONTEXT block."""
    return "\n\n".join(_format_chunk(ch) for ch in chunks)


def _history_messages(history: Optional[Sequence[Turn]]) -> List[dict]:
    """Recent prior turns as chat messages (answers truncated to bound the prompt)."""
    if not history:
        return []
    messages: List[dict] = []
    for question, answer in list(history)[-config.HISTORY_TURNS:]:
        q = (question or "").strip()
        a = (answer or "").strip()
        if not q:
            continue
        if len(a) > config.HISTORY_ANSWER_CHARS:
            a = a[: config.HISTORY_ANSWER_CHARS].rstrip() + " ..."
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages


def build_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Optional[Sequence[Turn]] = None,
    mode: str | None = None,
    allow_general: bool = False,
) -> List[dict]:
    """Assemble the messages list: mode-aware system prompt + history + context + question."""
    context = build_context(chunks) if chunks else "(no relevant context found)"
    user = f"CONTEXT:\n{context}\n\nUSER:\n{question}"
    return [
        {"role": "system", "content": config.system_prompt(mode, allow_general=allow_general)},
        *_history_messages(history),
        {"role": "user", "content": user},
    ]


def _default_chat(
    messages: List[dict], max_tokens: int, on_token: Callable[[str], None] | None
) -> str:
    from . import foundry_client

    return foundry_client.chat(messages, max_tokens=max_tokens, on_token=on_token)


def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Optional[Sequence[Turn]] = None,
    mode: str | None = None,
    chat_fn: Callable[[List[dict]], str] | None = None,
    on_token: Callable[[str], None] | None = None,
    allow_general: bool = False,
) -> str:
    """Build the grounded prompt and return the model's answer text.

    With no retrieved context there is nothing to ground on, so we refuse immediately rather
    than pay for a model call that should refuse anyway -- general knowledge is only offered
    when there IS relevant context to attempt first. `on_token`, if given, streams each text
    delta as it arrives (the refusal is emitted whole).
    """
    if not chunks:
        if on_token is not None:
            on_token(config.REFUSAL_TEXT)
        return config.REFUSAL_TEXT
    fitted = fit_context(chunks)
    messages = build_messages(
        question, fitted, history=history, mode=mode, allow_general=allow_general
    )
    if chat_fn is not None:
        answer = chat_fn(messages)
        if on_token is not None and answer:
            on_token(answer)  # fakes don't stream; emit the whole answer once
    else:
        answer = _default_chat(messages, config.mode_config(mode)["max_tokens"], on_token)
    return answer.strip() if answer else config.REFUSAL_TEXT
