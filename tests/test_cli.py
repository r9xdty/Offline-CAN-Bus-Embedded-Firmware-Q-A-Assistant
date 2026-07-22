"""Offline tests for the CLI loop and the Foundry client's error mapping.

No server or models needed: the pipeline, warm-up, and `input()` are faked.
"""

from __future__ import annotations

import builtins
import io
from contextlib import redirect_stdout

import pytest

from src import cli, foundry_client


class _FakeAnswer:
    def __init__(self, question="q", mode="short"):
        self.question = question
        self.answer = "The maximum length at 500 kbps is about 100 m. [can_2_0_basics.md]"
        self.sources = ["can_2_0_basics.md"]
        self.chunks: list = []
        self.elapsed_s = 0.5
        self.top_score = 0.72
        self.mode = mode


class _FakePipeline:
    """Records the (history, mode) of each call, for assertions. `last` = newest instance."""

    last: "_FakePipeline | None" = None

    def __init__(self, *args, **kwargs):
        self.calls = []
        _FakePipeline.last = self

    @property
    def size(self) -> int:
        return 5

    def answer(self, question, history=None, mode=None, k=3):
        self.calls.append({"question": question, "history": list(history or []), "mode": mode})
        return _FakeAnswer(question=question, mode=mode or "short")


def _fake_inputs(seq):
    it = iter(seq)
    return lambda prompt="": next(it)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_quit_prints_a_goodbye_message(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(builtins, "input", _fake_inputs(["quit"]))

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run(debug=False)
    out = buf.getvalue()
    assert "Chat model:" in out  # the active model is shown at startup
    assert any(msg in out for msg in cli.GOODBYE_MESSAGES)  # the fixed goodbye line ran


def test_answers_then_quits_on_q(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(builtins, "input", _fake_inputs(["what is the max bus length?", "q"]))

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run()
    out = buf.getvalue()
    assert "Sources: ['can_2_0_basics.md']" in out
    assert any(msg in out for msg in cli.GOODBYE_MESSAGES)


def test_warmup_failure_is_nonfatal(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    def boom():
        raise RuntimeError("server down")

    monkeypatch.setattr(cli.foundry_client, "warmup", boom)
    monkeypatch.setattr(builtins, "input", _fake_inputs(["quit"]))

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run()  # must not raise even if warm-up fails
    assert any(msg in buf.getvalue() for msg in cli.GOODBYE_MESSAGES)


def test_mode_toggle_and_memory(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(
        builtins, "input",
        _fake_inputs([":explain", "what is a PGN?", "explain that more", "quit"]),
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run(mode="short")

    calls = _FakePipeline.last.calls
    # The :explain command is not a query; the two questions are.
    assert [c["question"] for c in calls] == ["what is a PGN?", "explain that more"]
    # Mode switched to explain before the first query and stuck.
    assert all(c["mode"] == "explain" for c in calls)
    # The follow-up carried the first turn as memory.
    assert calls[0]["history"] == []
    assert len(calls[1]["history"]) == 1
    assert calls[1]["history"][0][0] == "what is a PGN?"


def test_reset_clears_memory(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(
        builtins, "input", _fake_inputs(["first question", ":reset", "second question", "quit"])
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run()

    calls = _FakePipeline.last.calls
    assert [c["question"] for c in calls] == ["first question", "second question"]
    assert calls[1]["history"] == []  # :reset wiped the memory before the second question


# --------------------------------------------------------------------------- #
# foundry_client._call error mapping
# --------------------------------------------------------------------------- #
class _APITimeoutError(Exception):
    """Stand-in whose class name contains 'timeout', like openai.APITimeoutError."""


def test_timeout_is_mapped_to_actionable_error():
    def fn():
        raise _APITimeoutError("Request timed out.")

    with pytest.raises(RuntimeError) as info:
        foundry_client._call(fn, "phi-4-mini-instruct-openvino-gpu")
    assert "RAG_CHAT_MODEL" in str(info.value)  # points to the fast-model switch


def test_not_loaded_triggers_load_and_retry(monkeypatch):
    monkeypatch.setattr(foundry_client, "_ensure_loaded", lambda m: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("Model 'x' is not loaded. Please load the model.")
        return "ok"

    assert foundry_client._call(fn, "x") == "ok"
    assert calls["n"] == 2


def test_is_timeout_detection():
    assert foundry_client._is_timeout(_APITimeoutError("x"))
    assert foundry_client._is_timeout(Exception("The request timed out."))
    assert not foundry_client._is_timeout(Exception("model not loaded"))
