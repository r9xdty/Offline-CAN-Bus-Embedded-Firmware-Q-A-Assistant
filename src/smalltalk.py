"""Lightweight, non-grounded replies for greetings and "what can you do?" style questions.

Deliberately kept OUT of the RAG pipeline — `pipeline.answer` only ever produces a grounded,
cited answer or the exact refusal. The interfaces (CLI, Streamlit) check `reply()` first so a
bare "hi" or "what can you do?" gets a friendly response instead of a refusal, with no model
call. Matching is on the whole normalized message (exact set membership), never a substring,
so real questions like "what can you tell me about CAN FD?" are never intercepted.
"""

from __future__ import annotations

import re
from typing import Optional

GREETINGS = {
    "hi", "hii", "hiya", "hey", "hello", "helo", "yo", "sup", "howdy",
    "good morning", "good afternoon", "good evening", "gm",
    # Turkish greetings
    "selam", "merhaba", "gunaydin", "günaydın", "iyi aksamlar", "iyi akşamlar",
}

THANKS = {
    "thanks", "thank you", "thank u", "thx", "ty", "cheers", "appreciated",
    "tesekkurler", "teşekkürler", "tesekkur ederim", "teşekkür ederim", "sagol", "sağ ol",
}

META = {
    "what can you do", "what do you do", "who are you", "what are you",
    "help", "what is this", "what can i ask", "what can i ask you",
    "capabilities", "how do you work", "what do you know",
}

_GREETING_REPLY = (
    "Hi! I'm an offline CAN-bus / embedded-firmware assistant. Ask me a question grounded in "
    'your documents — e.g. "What is the maximum CAN bus length at 500 kbps?"'
)
_THANKS_REPLY = "You're welcome! Ask me anything else about CAN bus or embedded firmware."
_META_REPLY = (
    "I answer CAN-bus / embedded-firmware questions grounded in your local documents, with "
    "source citations, and I say when something isn't in them. Topics in the starter corpus: "
    "CAN 2.0, CAN FD, error handling, STM32 bit timing, J1939, CANopen, transceivers — and you "
    "can upload your own PDFs. In the CLI, type :help for commands."
)


def _normalize(text: str) -> str:
    """Lowercase and strip surrounding punctuation/whitespace for exact matching."""
    return re.sub(r"[^\w\s]", "", (text or "").strip().lower()).strip()


def reply(question: str) -> Optional[str]:
    """Return a canned reply for a greeting / thanks / meta question, else None."""
    q = _normalize(question)
    if not q:
        return None
    if q in GREETINGS:
        return _GREETING_REPLY
    if q in THANKS:
        return _THANKS_REPLY
    if q in META:
        return _META_REPLY
    return None
