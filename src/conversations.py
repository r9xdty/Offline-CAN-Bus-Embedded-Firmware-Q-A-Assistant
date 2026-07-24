"""Multi-conversation persistence for the Streamlit UI (chat history survives restarts).

Pure data-layer module: every function takes/returns plain dicts (JSON-serializable) and does
its own I/O explicitly via a `path` argument — no globals, no Streamlit import — so it can be
unit-tested offline and reused if another UI ever needs it.

On-disk shape::

    {"conversations": [{"id": str, "title": str, "created_at": float, "messages": [...]}, ...],
     "current_id": str | None}

Each `messages` entry is one turn dict as already produced by `app_streamlit.py` (the same
shape rendered by `_render_turn`), stored as-is.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

DEFAULT_TITLE = "New chat"


def empty_state() -> dict:
    """A fresh, conversation-free store."""
    return {"conversations": [], "current_id": None}


def load(path: str | Path) -> dict:
    """Read the store from `path`; fall back to `empty_state()` on any problem.

    Never raises: a missing file, unreadable file, invalid JSON, or unexpected shape (e.g. an
    old format, or a hand-edited file) all just start the user fresh instead of crashing the app.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return empty_state()
    if not isinstance(data, dict) or not isinstance(data.get("conversations"), list):
        return empty_state()
    return data


def save(path: str | Path, data: dict) -> None:
    """Write the store to `path` as pretty-printed UTF-8 JSON, creating the parent dir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def new_conversation(data: dict, title: str = DEFAULT_TITLE) -> str:
    """Append a fresh, empty conversation, make it current, and return its id."""
    conv_id = uuid.uuid4().hex
    data["conversations"].append(
        {"id": conv_id, "title": title, "created_at": time.time(), "messages": []}
    )
    data["current_id"] = conv_id
    return conv_id


def current(data: dict) -> dict | None:
    """The conversation matching `current_id`, or None if there isn't one."""
    conv_id = data.get("current_id")
    for conv in data["conversations"]:
        if conv["id"] == conv_id:
            return conv
    return None


def ensure_current(data: dict) -> dict:
    """Guarantee a current conversation exists, creating one if needed, and return it."""
    conv = current(data)
    if conv is None:
        conv_id = new_conversation(data)
        conv = current(data)
        assert conv is not None and conv["id"] == conv_id  # new_conversation just added it
    return conv


def delete_conversation(data: dict, conv_id: str) -> None:
    """Remove a conversation; if it was current, re-point current_id at the most recent one left.

    No-op if `conv_id` isn't present.
    """
    conversations = data["conversations"]
    idx = next((i for i, c in enumerate(conversations) if c["id"] == conv_id), None)
    if idx is None:
        return
    was_current = data.get("current_id") == conv_id
    del conversations[idx]
    if was_current:
        if conversations:
            newest = max(conversations, key=lambda c: c["created_at"])
            data["current_id"] = newest["id"]
        else:
            data["current_id"] = None


def derive_title(messages: list[dict], max_len: int = 40) -> str:
    """The first non-small-talk question, trimmed to `max_len` chars, or the default title."""
    for m in messages:
        if m.get("smalltalk"):
            continue
        question = (m.get("question") or "").strip()
        if question:
            if len(question) > max_len:
                return question[: max_len - 1].rstrip() + "…"
            return question
    return DEFAULT_TITLE


def touch_title(conversation: dict) -> None:
    """Replace a still-default/empty title with one derived from the conversation's messages."""
    title = (conversation.get("title") or "").strip()
    if not title or title == DEFAULT_TITLE:
        conversation["title"] = derive_title(conversation["messages"])
