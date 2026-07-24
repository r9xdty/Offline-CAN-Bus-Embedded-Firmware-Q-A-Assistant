"""Offline tests for src/conversations.py (multi-chat persistence for the Streamlit UI).

Pure-function module, no Streamlit/Foundry involved — everything runs against a tmp_path
JSON file.
"""

from __future__ import annotations

from pathlib import Path

from src import conversations


# --------------------------------------------------------------------------- #
# load / save
# --------------------------------------------------------------------------- #
def test_load_missing_path_returns_empty_state(tmp_path: Path):
    assert conversations.load(tmp_path / "nope.json") == conversations.empty_state()


def test_load_corrupt_file_returns_empty_state_without_raising(tmp_path: Path):
    path = tmp_path / "chats.json"
    path.write_text("not { valid json at all", encoding="utf-8")
    assert conversations.load(path) == conversations.empty_state()


def test_load_wrong_shape_returns_empty_state(tmp_path: Path):
    path = tmp_path / "chats.json"
    path.write_text('{"foo": "bar"}', encoding="utf-8")
    assert conversations.load(path) == conversations.empty_state()


def test_save_then_load_round_trips(tmp_path: Path):
    path = tmp_path / "nested" / "chats.json"  # parent dir doesn't exist yet
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data, title="CAN bit timing")
    conv = conversations.current(data)
    conv["messages"].append(
        {
            "question": "What is the sample point?",
            "answer": "Typically 87.5% on the bit time.",
            "sources": ["can_fd_basics.md"],
            "top_score": 0.62,
            "elapsed_s": 1.2,
            "mode": "short",
            "chunks": [["can_fd_basics.md", 0, 0.62, "The sample point ..."]],
            "kind": "grounded",
        }
    )

    conversations.save(path, data)
    reloaded = conversations.load(path)

    assert reloaded["current_id"] == conv_id
    assert len(reloaded["conversations"]) == 1
    got = reloaded["conversations"][0]
    assert got["id"] == conv_id
    assert got["title"] == "CAN bit timing"
    assert len(got["messages"]) == 1
    assert got["messages"][0]["sources"] == ["can_fd_basics.md"]
    assert got["messages"][0]["chunks"][0][0] == "can_fd_basics.md"


# --------------------------------------------------------------------------- #
# new_conversation / current / ensure_current
# --------------------------------------------------------------------------- #
def test_new_conversation_sets_current_and_is_listed():
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data)
    assert data["current_id"] == conv_id
    assert any(c["id"] == conv_id for c in data["conversations"])


def test_new_conversation_ids_are_distinct():
    data = conversations.empty_state()
    id1 = conversations.new_conversation(data)
    id2 = conversations.new_conversation(data)
    assert id1 != id2


def test_current_returns_none_when_unset():
    data = conversations.empty_state()
    assert conversations.current(data) is None


def test_ensure_current_creates_one_on_empty_state():
    data = conversations.empty_state()
    conv = conversations.ensure_current(data)
    assert conv is not None
    assert conv["id"] == data["current_id"]
    assert len(data["conversations"]) == 1


def test_ensure_current_repairs_dangling_current_id():
    data = conversations.empty_state()
    data["current_id"] = "does-not-exist"
    conv = conversations.ensure_current(data)
    assert conv is not None
    assert conv["id"] == data["current_id"]


# --------------------------------------------------------------------------- #
# delete_conversation
# --------------------------------------------------------------------------- #
def test_delete_conversation_removes_it():
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data)
    conversations.delete_conversation(data, conv_id)
    assert data["conversations"] == []


def test_delete_current_reassigns_to_a_remaining_conversation():
    data = conversations.empty_state()
    older = conversations.new_conversation(data)
    data["conversations"][0]["created_at"] = 100.0
    newer = conversations.new_conversation(data)  # becomes current
    data["conversations"][1]["created_at"] = 200.0

    conversations.delete_conversation(data, newer)
    assert data["current_id"] == older


def test_delete_last_conversation_leaves_current_id_none_then_ensure_current_creates_fresh():
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data)
    conversations.delete_conversation(data, conv_id)
    assert data["current_id"] is None

    fresh = conversations.ensure_current(data)
    assert fresh is not None
    assert data["current_id"] == fresh["id"]


def test_delete_conversation_missing_id_is_safe_noop():
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data)
    conversations.delete_conversation(data, "not-a-real-id")
    assert data["current_id"] == conv_id
    assert len(data["conversations"]) == 1


def test_delete_non_current_leaves_current_id_untouched():
    data = conversations.empty_state()
    first = conversations.new_conversation(data)
    second = conversations.new_conversation(data)  # current
    conversations.delete_conversation(data, first)
    assert data["current_id"] == second
    assert len(data["conversations"]) == 1


# --------------------------------------------------------------------------- #
# derive_title / touch_title
# --------------------------------------------------------------------------- #
def test_derive_title_uses_first_non_smalltalk_question():
    messages = [
        {"question": "hi", "answer": "Hello!", "smalltalk": True},
        {"question": "What causes bus-off?", "answer": "..."},
        {"question": "second question", "answer": "..."},
    ]
    assert conversations.derive_title(messages) == "What causes bus-off?"


def test_derive_title_truncates_with_ellipsis():
    long_q = "x" * 60
    messages = [{"question": long_q, "answer": "..."}]
    title = conversations.derive_title(messages, max_len=40)
    assert len(title) == 40
    assert title.endswith("…")
    assert title[:-1] == "x" * 39


def test_derive_title_fallback_when_only_smalltalk():
    messages = [{"question": "hello there", "answer": "hi", "smalltalk": True}]
    assert conversations.derive_title(messages) == "New chat"


def test_derive_title_fallback_on_empty_messages():
    assert conversations.derive_title([]) == "New chat"


def test_touch_title_updates_default_title():
    conv = {"title": "New chat", "messages": [{"question": "What is a PGN?", "answer": "..."}]}
    conversations.touch_title(conv)
    assert conv["title"] == "What is a PGN?"


def test_touch_title_updates_empty_title():
    conv = {"title": "", "messages": [{"question": "What is a PDO?", "answer": "..."}]}
    conversations.touch_title(conv)
    assert conv["title"] == "What is a PDO?"


def test_touch_title_leaves_custom_title_unchanged():
    conv = {
        "title": "My custom chat name",
        "messages": [{"question": "What is a PGN?", "answer": "..."}],
    }
    conversations.touch_title(conv)
    assert conv["title"] == "My custom chat name"


# --------------------------------------------------------------------------- #
# rename_conversation
# --------------------------------------------------------------------------- #
def test_rename_conversation_sets_title_and_pins_it():
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data)
    conversations.rename_conversation(data, conv_id, "My renamed chat")
    conv = conversations.current(data)
    assert conv["title"] == "My renamed chat"
    assert conv["title_pinned"] is True


def test_rename_conversation_blank_title_falls_back_to_default():
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data, title="Something else")
    conversations.rename_conversation(data, conv_id, "   ")
    conv = conversations.current(data)
    assert conv["title"] == conversations.DEFAULT_TITLE
    assert conv["title_pinned"] is True


def test_rename_conversation_unknown_id_is_safe_noop():
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data, title="Original")
    conversations.rename_conversation(data, "not-a-real-id", "New name")
    conv = conversations.current(data)
    assert conv["id"] == conv_id
    assert conv["title"] == "Original"
    assert "title_pinned" not in conv


def test_touch_title_leaves_renamed_title_unchanged_after_new_question():
    data = conversations.empty_state()
    conv_id = conversations.new_conversation(data)
    conversations.rename_conversation(data, conv_id, "Pinned name")
    conv = conversations.current(data)
    conv["messages"].append({"question": "What causes bus-off?", "answer": "..."})
    conversations.touch_title(conv)
    assert conv["title"] == "Pinned name"
