"""Offline unit tests for the RAG pipeline.

These exercise every part that does NOT require the Foundry runtime — chunking, vector
serialization, cosine search, the SQLite round-trip, prompt assembly, context fitting, the
pipeline's refusal/citation logic, and the eval grader — by injecting deterministic fake
embed/chat functions. They run on any machine with `pytest`, no GPU or models needed.

The parts that genuinely need the local LLM (model-generated facts and out-of-corpus refusals)
are covered by `tests/run_eval.py` against the real pipeline on the target machine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List

import numpy as np
import pytest

from src import config, generate, ingest, vectors
from src.pipeline import Pipeline
from src.retrieve import Retriever, RetrievedChunk, cosine_top_k
from tests.run_eval import grade, run_eval, load_eval

RAW_DIR = config.RAW_DIR


# --------------------------------------------------------------------------- #
# Deterministic fake embedder: bag-of-words over a fixed vocabulary.
# --------------------------------------------------------------------------- #
VOCAB = ["length", "identifier", "error", "timing", "canopen", "j1939", "transceiver", "fd"]


def _bow_vector(text: str) -> np.ndarray:
    low = text.lower()
    vec = np.array([low.count(w) for w in VOCAB], dtype=np.float32)
    if not vec.any():
        vec = vec + 1e-3  # avoid an all-zero vector so normalization is well-defined
    return vec


def bow_embed(texts: List[str]) -> np.ndarray:
    return np.vstack([_bow_vector(t) for t in texts]).astype(np.float32)


def bow_embed_query(text: str) -> np.ndarray:
    return _bow_vector(text)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_chunk_text_respects_max_and_is_nonempty():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(8))
    chunks = ingest.chunk_text(text, target=700, max_chars=800, overlap=100)
    assert chunks, "expected at least one chunk"
    for ch in chunks:
        assert ch.strip(), "no whitespace-only chunks"
        # Allow a little slack for the joined overlap carry, but stay near the max.
        assert len(ch) <= 800 + 100


def test_chunk_text_short_input_single_chunk():
    chunks = ingest.chunk_text("A short note about CAN bus termination.")
    assert chunks == ["A short note about CAN bus termination."]


def test_chunk_text_has_overlap_between_consecutive_chunks():
    # Build text long enough to force multiple chunks.
    sentences = [f"Sentence number {i} about bit timing segments and quanta." for i in range(60)]
    text = " ".join(sentences)
    chunks = ingest.chunk_text(text, target=400, max_chars=500, overlap=80)
    assert len(chunks) >= 2
    # Some token from the tail of chunk n should reappear at the head of chunk n+1.
    tail_words = set(chunks[0].split()[-6:])
    head_words = set(chunks[1].split()[:12])
    assert tail_words & head_words, "consecutive chunks should share overlap text"


def test_real_corpus_chunks_are_reasonable():
    records = ingest.build_records(RAW_DIR)
    assert len(records) >= 10, "corpus should yield a healthy number of chunks"
    sources = {src for (src, _, _) in records}
    assert "can_2_0_basics.md" in sources
    for _, _, content in records:
        assert content.strip()
        assert len(content) <= config.CHUNK_MAX_CHARS + config.CHUNK_OVERLAP_CHARS + 50


# --------------------------------------------------------------------------- #
# Vectors
# --------------------------------------------------------------------------- #
def test_l2_normalize_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = vectors.l2_normalize(v)
    assert np.isclose(np.linalg.norm(n), 1.0)


def test_blob_round_trip():
    v = vectors.l2_normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    back = vectors.from_blob(vectors.to_blob(v))
    assert np.allclose(v, back)


def test_zero_vector_normalize_is_safe():
    z = np.zeros(4, dtype=np.float32)
    assert np.allclose(vectors.l2_normalize(z), z)


# --------------------------------------------------------------------------- #
# Cosine search
# --------------------------------------------------------------------------- #
def test_cosine_top_k_orders_by_similarity():
    matrix = np.array(
        [
            vectors.l2_normalize(np.array([1.0, 0.0, 0.0])),
            vectors.l2_normalize(np.array([0.0, 1.0, 0.0])),
            vectors.l2_normalize(np.array([0.9, 0.1, 0.0])),
        ],
        dtype=np.float32,
    )
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    top = cosine_top_k(q, matrix, k=2)
    assert [i for i, _ in top] == [0, 2]  # exact match first, then the near one
    assert top[0][1] >= top[1][1]


def test_cosine_top_k_empty_matrix():
    assert cosine_top_k(np.array([1.0]), np.empty((0, 0), dtype=np.float32), k=3) == []


# --------------------------------------------------------------------------- #
# Ingest + retrieve round trip (fake embedder, real SQLite)
# --------------------------------------------------------------------------- #
def test_ingest_and_retrieve_round_trip(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "length_doc.md").write_text("CAN bus length depends on bit rate. length length length")
    (raw / "timing_doc.md").write_text("Bit timing uses time quanta and segments. timing timing timing")
    (raw / "error_doc.md").write_text("Error confinement uses counters. error error error")

    db = tmp_path / "kb.sqlite"
    count = ingest.ingest(raw_dir=raw, db_path=db, embed_fn=bow_embed)
    assert count == 3

    retriever = Retriever(db_path=db, embed_query_fn=bow_embed_query)
    assert retriever.size == 3

    hits = retriever.retrieve("tell me about bus timing", k=1)
    assert hits[0].source == "timing_doc.md"

    hits = retriever.retrieve("what about error handling", k=1)
    assert hits[0].source == "error_doc.md"


def test_retriever_handles_missing_kb_gracefully(tmp_path: Path):
    # A DB file that was never ingested (no `chunks` table) should look like an empty KB.
    db = tmp_path / "never_ingested.sqlite"
    retriever = Retriever(db_path=db, embed_query_fn=bow_embed_query)
    assert retriever.size == 0
    assert retriever.retrieve("anything", k=3) == []


def test_write_chunks_is_idempotent(tmp_path: Path):
    db = tmp_path / "kb.sqlite"
    records = [("a.md", 0, "length length"), ("b.md", 0, "timing timing")]
    emb = bow_embed([c for _, _, c in records])
    ingest.write_chunks(db, records, emb)
    n2 = ingest.write_chunks(db, records, emb)  # rebuild
    assert n2 == 2
    retriever = Retriever(db_path=db, embed_query_fn=bow_embed_query)
    assert retriever.size == 2  # not duplicated


# --------------------------------------------------------------------------- #
# Generation: prompt assembly + context fitting
# --------------------------------------------------------------------------- #
def _chunk(source: str, content: str, score: float = 0.9, idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(id=idx, source=source, chunk_index=idx, content=content, score=score)


def test_build_messages_structure():
    chunks = [_chunk("can_fd_basics.md", "CAN FD supports up to 64 data bytes.")]
    messages = generate.build_messages("How many bytes?", chunks)
    assert messages[0]["role"] == "system"
    assert config.REFUSAL_TEXT in messages[0]["content"]
    user = messages[1]["content"]
    assert "[can_fd_basics.md]" in user  # source label present for citation
    assert "How many bytes?" in user
    assert "CONTEXT:" in user and "USER:" in user


def test_fit_context_truncates_to_budget():
    big = "x" * 300
    chunks = [_chunk(f"doc{i}.md", big, score=1.0 - i * 0.1, idx=i) for i in range(5)]
    kept = generate.fit_context(chunks, char_budget=700)
    assert 1 <= len(kept) < len(chunks)  # dropped the ones that don't fit
    # Highest-scored chunks are kept first.
    assert kept[0].source == "doc0.md"


def test_generate_answer_refuses_without_chunks():
    assert generate.generate_answer("anything", []) == config.REFUSAL_TEXT


def test_generate_answer_uses_chat_fn():
    chunks = [_chunk("can_2_0_basics.md", "Max length at 500 kbps is about 100 meters.")]

    def fake_chat(messages: List[dict]) -> str:
        # A grounded extractive stub that cites its source.
        return "About 100 meters. [can_2_0_basics.md]"

    out = generate.generate_answer("length at 500 kbps?", chunks, chat_fn=fake_chat)
    assert "100 meters" in out


def test_generate_answer_system_prompt_reflects_allow_general():
    chunks = [_chunk("can_2_0_basics.md", "Max length at 500 kbps is about 100 meters.")]
    captured: dict = {}

    def fake_chat(messages: List[dict]) -> str:
        captured["system"] = messages[0]["content"]
        return "About 100 meters. [can_2_0_basics.md]"

    generate.generate_answer("q", chunks, chat_fn=fake_chat, allow_general=True)
    assert config.GENERAL_LABEL in captured["system"]

    generate.generate_answer("q", chunks, chat_fn=fake_chat, allow_general=False)
    assert config.GENERAL_LABEL not in captured["system"]


# --------------------------------------------------------------------------- #
# Answer modes + conversation memory
# --------------------------------------------------------------------------- #
def test_build_messages_includes_history_and_mode():
    chunks = [_chunk("can_fd_basics.md", "CAN FD supports up to 64 data bytes.")]
    history = [("What is CAN FD?", "A flexible-data-rate extension of CAN.")]
    messages = generate.build_messages("How many bytes?", chunks, history=history, mode="explain")

    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]  # system, one prior turn, current
    assert config.ANSWER_MODES["explain"]["instruction"] in messages[0]["content"]
    assert config.REFUSAL_TEXT in messages[0]["content"]  # refusal rule holds in explain mode
    assert messages[1]["content"] == "What is CAN FD?"
    assert "How many bytes?" in messages[-1]["content"]


def test_short_and_explain_use_different_instructions():
    chunks = [_chunk("can_fd_basics.md", "CAN FD supports up to 64 data bytes.")]
    short_sys = generate.build_messages("q", chunks, mode="short")[0]["content"]
    explain_sys = generate.build_messages("q", chunks, mode="explain")[0]["content"]
    assert config.ANSWER_MODES["short"]["instruction"] in short_sys
    assert config.ANSWER_MODES["explain"]["instruction"] in explain_sys
    assert short_sys != explain_sys


def test_history_answers_are_truncated(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_ANSWER_CHARS", 20)
    chunks = [_chunk("a.md", "x")]
    long_answer = "y" * 200
    messages = generate.build_messages("q", chunks, history=[("prev?", long_answer)])
    prior_assistant = messages[2]["content"]
    assert len(prior_assistant) < 200 and prior_assistant.endswith("...")


def test_retrieval_query_expands_elliptical_followups():
    from src.pipeline import _retrieval_query

    history = [("What is the max CAN bus length at 500 kbps?", "About 100 m.")]
    # Elliptical follow-up gets the previous question prepended for retrieval.
    assert "500 kbps" in _retrieval_query("what about at 250 kbps?", history)
    assert "500 kbps" in _retrieval_query("explain that", history)
    # A self-contained question is used as-is (no dilution).
    assert _retrieval_query("What is a PGN in J1939 and how is it structured?", history) == (
        "What is a PGN in J1939 and how is it structured?"
    )
    # No history -> unchanged.
    assert _retrieval_query("explain that", None) == "explain that"


def test_answer_threads_mode_and_reports_it(tmp_path):
    pipe = _pipeline_with(tmp_path, lambda m: "About 100 meters. [length_doc.md]")
    ans = pipe.answer("length at 500 kbps", mode="explain")
    assert ans.mode == "explain"


def test_generate_answer_emits_tokens(monkeypatch):
    chunks = [_chunk("a.md", "content")]
    tokens = []
    out = generate.generate_answer(
        "q", chunks, chat_fn=lambda m: "hello answer [a.md]", on_token=tokens.append
    )
    assert tokens == ["hello answer [a.md]"]  # fake emits the whole answer once
    assert out == "hello answer [a.md]"


def test_generate_answer_streams_refusal_without_chunks():
    tokens = []
    out = generate.generate_answer("q", [], on_token=tokens.append)
    assert out == config.REFUSAL_TEXT
    assert tokens == [config.REFUSAL_TEXT]  # refusal is streamed too


def test_pipeline_answer_streams(tmp_path):
    tokens = []
    pipe = _pipeline_with(tmp_path, lambda m: "About 100 meters. [length_doc.md]")
    ans = pipe.answer("length at 500 kbps", on_token=tokens.append)
    assert "".join(tokens) == ans.answer  # streamed content matches the final answer


# --------------------------------------------------------------------------- #
# Pipeline: refusal + citation logic
# --------------------------------------------------------------------------- #
def _pipeline_with(tmp_path: Path, chat_fn: Callable[[List[dict]], str]) -> Pipeline:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "length_doc.md").write_text("CAN bus length at 500 kbps is about 100 meters. length length")
    (raw / "timing_doc.md").write_text("Bit timing uses SYNC_SEG BS1 BS2. timing timing")
    db = tmp_path / "kb.sqlite"
    ingest.ingest(raw_dir=raw, db_path=db, embed_fn=bow_embed)
    return Pipeline(db_path=db, embed_query_fn=bow_embed_query, chat_fn=chat_fn)


def test_pipeline_empty_question_refuses(tmp_path: Path):
    pipe = _pipeline_with(tmp_path, lambda m: "should not be called")
    ans = pipe.answer("   ")
    assert ans.is_refusal
    assert ans.sources == []


def test_pipeline_grounded_answer_has_sources(tmp_path: Path):
    def chat(messages: List[dict]) -> str:
        return "The maximum length is about 100 meters. [length_doc.md]"

    pipe = _pipeline_with(tmp_path, chat)
    ans = pipe.answer("length at 500 kbps")
    assert not ans.is_refusal
    assert "length_doc.md" in ans.sources
    assert ans.chunks and ans.chunks[0].source == "length_doc.md"


def test_pipeline_refusal_string_clears_sources(tmp_path: Path):
    pipe = _pipeline_with(tmp_path, lambda m: config.REFUSAL_TEXT)
    ans = pipe.answer("something the model cannot answer")
    assert ans.is_refusal
    assert ans.sources == []


def test_similarity_threshold_refuses_without_calling_the_model(tmp_path, monkeypatch):
    def chat_must_not_run(messages):
        raise AssertionError("chat must not run when every chunk is below the score floor")

    pipe = _pipeline_with(tmp_path, chat_must_not_run)
    # A cosine floor above 1.0 filters everything -> deterministic refusal, no model call.
    monkeypatch.setattr(config, "MIN_SCORE", 2.0)
    ans = pipe.answer("length at 500 kbps")
    assert ans.is_refusal
    assert ans.sources == []
    assert ans.chunks  # retrieved chunks are still surfaced (below-threshold) for transparency


def test_answer_reports_latency_and_top_score(tmp_path):
    pipe = _pipeline_with(tmp_path, lambda m: "About 100 meters. [length_doc.md]")
    ans = pipe.answer("length at 500 kbps")
    assert ans.elapsed_s >= 0.0
    assert ans.top_score is not None and ans.top_score > 0


def test_pipeline_general_knowledge_answer_is_labeled_and_uncited(tmp_path, monkeypatch):
    # Open the domain gate unconditionally so any relevant question qualifies.
    monkeypatch.setattr(config, "DOMAIN_SCORE", 0.0)

    def chat(messages: List[dict]) -> str:
        return config.GENERAL_LABEL + " CAN uses differential signaling."

    pipe = _pipeline_with(tmp_path, chat)
    ans = pipe.answer("length at 500 kbps")
    assert ans.kind == "general"
    assert ans.is_general
    assert ans.sources == []
    assert config.GENERAL_LABEL in ans.answer


def test_pipeline_general_knowledge_disabled_falls_back(tmp_path, monkeypatch):
    # Even with the domain gate wide open, the master switch off must restore old behavior.
    monkeypatch.setattr(config, "GENERAL_KNOWLEDGE_ENABLED", False)
    monkeypatch.setattr(config, "DOMAIN_SCORE", 0.0)

    pipe = _pipeline_with(tmp_path, lambda m: config.REFUSAL_TEXT)
    ans = pipe.answer("length at 500 kbps")
    assert ans.is_refusal
    assert ans.kind == "refusal"


def test_pipeline_domain_gate_keeps_strict_prompt_when_below_floor(tmp_path, monkeypatch):
    # An unreachable domain floor means allow_general stays False -> the strict prompt (no
    # general-knowledge instructions) must be what's actually sent to the model.
    monkeypatch.setattr(config, "DOMAIN_SCORE", 2.0)

    def chat(messages: List[dict]) -> str:
        assert config.GENERAL_LABEL not in messages[0]["content"]
        return "The maximum length is about 100 meters. [length_doc.md]"

    pipe = _pipeline_with(tmp_path, chat)
    ans = pipe.answer("length at 500 kbps")
    assert ans.kind == "grounded"
    assert "length_doc.md" in ans.sources


def test_pipeline_general_enabled_override_disables(tmp_path, monkeypatch):
    # Env flag says general knowledge is on, but a per-call override of False must win.
    monkeypatch.setattr(config, "GENERAL_KNOWLEDGE_ENABLED", True)
    monkeypatch.setattr(config, "DOMAIN_SCORE", 0.0)
    captured: dict = {}

    def chat(messages: List[dict]) -> str:
        captured["system"] = messages[0]["content"]
        return config.GENERAL_LABEL + " CAN uses differential signaling."

    pipe = _pipeline_with(tmp_path, chat)
    pipe.answer("length at 500 kbps", general_enabled=False)
    assert config.GENERAL_LABEL not in captured["system"]


def test_pipeline_general_enabled_override_enables(tmp_path, monkeypatch):
    # Env flag says general knowledge is off, but a per-call override of True must win.
    monkeypatch.setattr(config, "GENERAL_KNOWLEDGE_ENABLED", False)
    monkeypatch.setattr(config, "DOMAIN_SCORE", 0.0)
    captured: dict = {}

    def chat(messages: List[dict]) -> str:
        captured["system"] = messages[0]["content"]
        return "The maximum length is about 100 meters. [length_doc.md]"

    pipe = _pipeline_with(tmp_path, chat)
    pipe.answer("length at 500 kbps", general_enabled=True)
    assert config.GENERAL_LABEL in captured["system"]


def test_pdf_citation_is_recognized(tmp_path):
    db = tmp_path / "kb.sqlite"
    # A PDF uploaded via the UI is stored by add_document with a .pdf source label.
    ingest.add_document(
        "datasheet.pdf",
        "length length length CAN bus length at 500 kbps 100 meters",
        db,
        bow_embed,
    )

    pipe = Pipeline(
        db_path=db,
        embed_query_fn=bow_embed_query,
        chat_fn=lambda m: "About 100 meters. [datasheet.pdf]",
    )
    ans = pipe.answer("length at 500 kbps")
    assert ans.sources == ["datasheet.pdf"]  # .pdf citation is now parsed


# --------------------------------------------------------------------------- #
# Eval harness grading logic
# --------------------------------------------------------------------------- #
def test_grade_refusal_pass_and_fail():
    item = {"expected_behavior": "refuse"}
    ok, _ = grade(item, config.REFUSAL_TEXT, [])
    assert ok
    ok, _ = grade(item, "Here is a made-up answer.", [])
    assert not ok


def test_grade_answer_keyword_match():
    item = {"expected_behavior": "answer", "must_include_any": ["100"], "expect_source": "d.md"}
    ok, _ = grade(item, "about 100 meters", ["d.md"])
    assert ok
    ok, _ = grade(item, "no number here", ["d.md"])
    assert not ok
    ok, _ = grade(item, config.REFUSAL_TEXT, [])
    assert not ok  # refusing an answerable question fails


def test_eval_set_is_wellformed():
    items = load_eval()
    assert len(items) >= 10
    for it in items:
        assert it["expected_behavior"] in {"answer", "refuse"}
        if it["expected_behavior"] == "answer":
            assert it.get("must_include_any"), it["question"]


class _Ans:
    def __init__(self, answer, sources):
        self.answer = answer
        self.sources = sources


def test_run_eval_with_fake_answerer():
    """Drive the full eval harness with a stub answerer to prove scoring end-to-end."""
    items = load_eval()

    def fake_answer(question, history=None, mode=None) -> _Ans:
        item = next((i for i in items if i["question"] == question), None)
        if item is None:
            return _Ans("prior-turn context answer", [])  # a history (setup) question
        if item["expected_behavior"] == "refuse":
            return _Ans(config.REFUSAL_TEXT, [])
        needle = item["must_include_any"][0]
        src = item.get("expect_source", "")
        return _Ans(f"Grounded answer containing {needle}. [{src}]", [src] if src else [])

    report = run_eval(answer_fn=fake_answer, items=items)
    assert report["passed"] == report["total"]


def test_run_eval_multiturn_builds_history_and_passes_mode():
    items = [
        {
            "history_questions": ["Q1"],
            "question": "Q2",
            "expected_behavior": "answer",
            "must_include_any": ["ok"],
        }
    ]
    seen = []

    def fake_answer(question, history=None, mode=None) -> _Ans:
        seen.append((question, list(history or []), mode))
        return _Ans("ok answer", [])

    report = run_eval(answer_fn=fake_answer, items=items, mode="explain")
    # Q1 asked first with empty history; Q2 then asked with [(Q1, its answer)].
    assert seen[0] == ("Q1", [], "explain")
    assert seen[1][0] == "Q2"
    assert seen[1][1] == [("Q1", "ok answer")]
    assert report["passed"] == 1
    assert report["results"][0]["multi_turn"] is True
    assert report["mode"] == "explain"


def test_eval_to_markdown_renders_table():
    from tests.run_eval import to_markdown

    report = {
        "total": 1, "passed": 1, "mode": "short",
        "results": [{"question": "q?", "multi_turn": False, "passed": True,
                     "reason": "answer contains a required fact", "latency_s": 1.2}],
    }
    md = to_markdown(report)
    assert "1/1 passed" in md and "| ✅ |" in md and "mode: short" in md
