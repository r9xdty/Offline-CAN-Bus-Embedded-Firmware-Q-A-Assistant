"""Offline tests for the greeting / small-talk shortcut."""

from __future__ import annotations

import pytest

from src import smalltalk


@pytest.mark.parametrize("text", ["hi", "Hi", "HELLO", "hey!", "  merhaba  ", "good morning"])
def test_greetings_get_a_reply(text):
    reply = smalltalk.reply(text)
    assert reply is not None and "assistant" in reply.lower()


@pytest.mark.parametrize("text", ["thanks", "Thank you!", "teşekkürler"])
def test_thanks_get_a_reply(text):
    assert smalltalk.reply(text) == smalltalk._THANKS_REPLY


@pytest.mark.parametrize("text", ["what can you do", "What can you do?", "who are you", "help"])
def test_meta_questions_get_capabilities(text):
    assert smalltalk.reply(text) == smalltalk._META_REPLY


@pytest.mark.parametrize(
    "text",
    [
        "what can you tell me about CAN FD?",   # contains 'what can you' but is a real question
        "hi there, what is a PGN?",             # starts with 'hi' but is a real question
        "What is the maximum CAN bus length at 500 kbps?",
        "explain that",
        "thanks to the transceiver, what happens?",
    ],
)
def test_real_questions_are_not_intercepted(text):
    assert smalltalk.reply(text) is None  # exact match only, never substring


def test_empty_input_returns_none():
    assert smalltalk.reply("") is None
    assert smalltalk.reply("   ") is None
    assert smalltalk.reply("!!!") is None
