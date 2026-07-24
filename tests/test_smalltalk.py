"""Offline tests for the greeting / small-talk shortcut."""

from __future__ import annotations

import pytest

from src import smalltalk


@pytest.mark.parametrize(
    "text",
    [
        "hi", "Hi", "HELLO", "hey!", "  good morning  ",
        "hey there", "Hello there!", "morning", "Good day", "greetings",
    ],
)
def test_greetings_get_a_reply(text):
    reply = smalltalk.reply(text)
    assert reply is not None and "assistant" in reply.lower()


@pytest.mark.parametrize(
    "text",
    [
        "thanks", "Thank you!", "thanks a lot", "many thanks", "Thank you so much!",
        "thanks so much", "much appreciated", "thanks a bunch",
    ],
)
def test_thanks_get_a_reply(text):
    assert smalltalk.reply(text) == smalltalk._THANKS_REPLY


@pytest.mark.parametrize(
    "text",
    [
        "what can you do", "What can you do?", "who are you", "help",
        "what topics do you cover", "what can you help with", "What can you help me with?",
        "how does this work", "how do I use this", "what are your capabilities",
    ],
)
def test_meta_questions_get_capabilities(text):
    assert smalltalk.reply(text) == smalltalk._META_REPLY


@pytest.mark.parametrize(
    "text",
    ["how are you", "How are you?", "how are you doing", "how r u", "hows it going", "how's it going", "how is it going"],
)
def test_howareyou_get_a_reply(text):
    assert smalltalk.reply(text) == smalltalk._HOWAREYOU_REPLY


@pytest.mark.parametrize(
    "text",
    [
        "what can you tell me about CAN FD?",   # contains 'what can you' but is a real question
        "hi there, what is a PGN?",             # starts with 'hi' but is a real question
        "What is the maximum CAN bus length at 500 kbps?",
        "explain that",
        "thanks to the transceiver, what happens?",
        "how are you handling bus-off errors?",  # starts with 'how are you' but is a real question
        "what's up with the CRC in CAN FD?",      # starts with 'what's up' but is a real question
    ],
)
def test_real_questions_are_not_intercepted(text):
    assert smalltalk.reply(text) is None  # exact match only, never substring


def test_empty_input_returns_none():
    assert smalltalk.reply("") is None
    assert smalltalk.reply("   ") is None
    assert smalltalk.reply("!!!") is None
