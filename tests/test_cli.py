"""Offline tests for the CLI loop and the Foundry client's error mapping.

No server or models needed: the pipeline, warm-up, and `input()` are faked.
"""

from __future__ import annotations

import builtins
import io
from contextlib import redirect_stdout

import pytest

from src import cli, config, foundry_client


class _FakeAnswer:
    def __init__(self, question="q", mode="short"):
        self.question = question
        self.answer = "The maximum length at 500 kbps is about 100 m. [can_2_0_basics.md]"
        self.sources = ["can_2_0_basics.md"]
        self.chunks: list = []
        self.elapsed_s = 0.5
        self.top_score = 0.72
        self.mode = mode
        self.kind = "grounded"


class _FakeGeneralAnswer(_FakeAnswer):
    """A general-knowledge (uncited, unsourced) answer, for exercising the amber notice."""

    def __init__(self, question="q", mode="short"):
        super().__init__(question=question, mode=mode)
        self.answer = f"{config.GENERAL_LABEL} Termination resistors are typically 120 ohms."
        self.sources = []
        self.kind = "general"


class _FakePipeline:
    """Records the (history, mode) of each call, for assertions. `last` = newest instance."""

    last: "_FakePipeline | None" = None
    answer_cls = _FakeAnswer

    def __init__(self, *args, **kwargs):
        self.calls = []
        _FakePipeline.last = self

    @property
    def size(self) -> int:
        return 5

    def answer(self, question, history=None, mode=None, k=3, on_token=None, general_enabled=None):
        self.calls.append({
            "question": question,
            "history": list(history or []),
            "mode": mode,
            "general_enabled": general_enabled,
        })
        ans = self.answer_cls(question=question, mode=mode or "short")
        if on_token is not None:
            on_token(ans.answer)  # simulate streaming the whole answer
        return ans


class _FakeGeneralPipeline(_FakePipeline):
    """Like _FakePipeline, but every answer is a general-knowledge one."""

    answer_cls = _FakeGeneralAnswer


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


def test_greeting_is_not_sent_to_the_pipeline(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(builtins, "input", _fake_inputs(["hi", "quit"]))

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run()
    out = buf.getvalue()
    assert "assistant" in out.lower()  # the friendly greeting reply was printed
    assert _FakePipeline.last.calls == []  # a greeting never reached the RAG pipeline


def test_examples_command_lists_sample_questions(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(builtins, "input", _fake_inputs([":examples", "quit"]))

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run()
    out = buf.getvalue()
    assert config.EXAMPLE_QUESTIONS[0] in out  # at least the first sample question is listed
    assert _FakePipeline.last.calls == []  # :examples never reaches the pipeline


def test_general_answer_shows_notice_and_kind(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakeGeneralPipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(builtins, "input", _fake_inputs(["what resistor value is used?", "quit"]))

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run()
    out = buf.getvalue()
    assert "General knowledge — not grounded in your documents" in out
    assert "general" in out  # the kind token appears in the stats line


def test_general_command_toggles_and_is_passed_to_pipeline(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(
        builtins, "input",
        _fake_inputs([":general off", "a question", ":general on", "another", "quit"]),
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run()
    out = buf.getvalue()

    calls = _FakePipeline.last.calls
    assert [c["question"] for c in calls] == ["a question", "another"]
    assert calls[0]["general_enabled"] is False
    assert calls[1]["general_enabled"] is True
    assert "[general knowledge: off]" in out
    assert "[general knowledge: on]" in out


def test_streaming_prints_answer_once(monkeypatch):
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)
    monkeypatch.setattr(cli.foundry_client, "warmup", lambda: None)
    monkeypatch.setattr(builtins, "input", _fake_inputs(["a question", "quit"]))

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.run(stream=True)  # streaming path
    out = buf.getvalue()
    answer = _FakeAnswer().answer
    assert out.count(answer) == 1  # streamed via callback, not also printed separately
    assert "Sources: ['can_2_0_basics.md']" in out


# --------------------------------------------------------------------------- #
# foundry_client streaming
# --------------------------------------------------------------------------- #
class _FakeStreamClient:
    """Minimal OpenAI-shaped client whose chat completion streams fixed deltas."""

    class _Delta:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.delta = _FakeStreamClient._Delta(content)

    class _Chunk:
        def __init__(self, content):
            self.choices = [_FakeStreamClient._Choice(content)]

    class _Completions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return [
                _FakeStreamClient._Chunk("The max "),
                _FakeStreamClient._Chunk("is 100 m."),
                _FakeStreamClient._Chunk(None),  # empty delta must be skipped
            ]

    class _Chat:
        completions = None

    def __init__(self):
        self.chat = _FakeStreamClient._Chat()
        self.chat.completions = _FakeStreamClient._Completions()


def test_foundry_chat_streams_deltas(monkeypatch):
    monkeypatch.setattr(foundry_client, "get_client", lambda: _FakeStreamClient())
    tokens = []
    full = foundry_client.chat([{"role": "user", "content": "q"}], on_token=tokens.append)
    assert tokens == ["The max ", "is 100 m."]  # None delta skipped
    assert full == "The max is 100 m."


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
