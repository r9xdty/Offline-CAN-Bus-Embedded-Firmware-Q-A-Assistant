"""Central configuration: model IDs, chunk params, retrieval + context limits, paths.

All tunables live here so the pipeline modules stay declarative. The model IDs are the
**full variant IDs** from the build spec, pinned so device placement is deterministic:

- Chat runs on the Intel Iris Xe iGPU via OpenVINO (shared system RAM, does not touch the
  4 GB NVIDIA VRAM).
- Embedding runs on the NVIDIA RTX 3050 Ti via CUDA.

Because the two models sit on different processors there is no 4 GB VRAM contention.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = parent of this src/ directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.environ.get("RAG_DB_PATH", DATA_DIR / "kb.sqlite"))

# --------------------------------------------------------------------------- #
# Foundry Local model placement (full variant IDs — do not shorten)
# --------------------------------------------------------------------------- #
APP_NAME = "rag-can-assistant"

# Chat / generation — Intel iGPU (OpenVINO), shared RAM. Chosen after testing: refuses
# out-of-context questions correctly, stays concise. Its first load compiles the model on the
# iGPU and can be slow; to use the fast cached NVIDIA/TensorRT model instead, override without
# editing code:  setx RAG_CHAT_MODEL phi-3.5-mini-instruct-trtrtx-gpu
CHAT_MODEL_ID = os.environ.get("RAG_CHAT_MODEL", "phi-4-mini-instruct-openvino-gpu").strip()

# Embedding — NVIDIA RTX 3050 Ti (CUDA). Override with RAG_EMBED_MODEL (e.g. "...-generic-cpu").
EMBED_MODEL_ID = os.environ.get("RAG_EMBED_MODEL", "qwen3-embedding-0.6b-cuda-gpu").strip()

# Timeout (seconds) for requests to the Foundry server. Generous by default because the first
# iGPU/OpenVINO chat load compiles the model; bounded so a genuine hang surfaces as an error
# instead of freezing forever. Override with RAG_REQUEST_TIMEOUT.
REQUEST_TIMEOUT = float(os.environ.get("RAG_REQUEST_TIMEOUT", "300"))

# Foundry Local server endpoint. We talk to the running `foundry server` over its
# OpenAI-compatible HTTP endpoint (the in-process SDK core does not share the daemon's model
# cache / execution-provider packs). Leave blank to auto-discover from `foundry server status`;
# override by setting the FOUNDRY_LOCAL_ENDPOINT env var to the Web URL it prints
# (e.g. http://127.0.0.1:54163).
FOUNDRY_ENDPOINT = os.environ.get("FOUNDRY_LOCAL_ENDPOINT", "").strip()

# --------------------------------------------------------------------------- #
# Chunking (see spec §8.1) — ~500-800 chars with ~100 char overlap.
# --------------------------------------------------------------------------- #
CHUNK_TARGET_CHARS = 700
CHUNK_MAX_CHARS = 800
CHUNK_MIN_CHARS = 500
CHUNK_OVERLAP_CHARS = 100

# --------------------------------------------------------------------------- #
# Retrieval + generation
# --------------------------------------------------------------------------- #
TOP_K = 3

# Minimum cosine similarity for a retrieved chunk to be fed to the model. Chunks below this
# are dropped, so an off-topic question reaches the model with no context -> a clean refusal
# instead of a guess over weak matches. Conservative default; watch scores with `--debug` and
# raise it (e.g. 0.25-0.35) for stricter refusals, or set 0 to disable. Override: RAG_MIN_SCORE.
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.1"))

# Keep the whole prompt inside the 4K context budget (spec §2, §8.2).
CONTEXT_TOKEN_LIMIT = 4096
MAX_ANSWER_TOKENS = 512
# Tokens reserved for the system prompt + question + formatting scaffolding.
PROMPT_OVERHEAD_TOKENS = 320
# Rough chars-per-token estimate for budgeting (English prose ~4 chars/token).
CHARS_PER_TOKEN = 4

# Low temperature: grounded RAG wants deterministic, faithful answers over creativity.
TEMPERATURE = 0.1

# The exact string the model must emit when the answer is not in the corpus (spec §9).
REFUSAL_TEXT = "I don't have that information in the provided documents."

# --------------------------------------------------------------------------- #
# Answer modes: same grounded/refusal contract, different style + length.
# --------------------------------------------------------------------------- #
ANSWER_MODES = {
    "short": {
        "label": "Short",
        "max_tokens": 160,
        "instruction": "Answer style: give the direct, definite answer in one or two sentences. No preamble.",
    },
    "explain": {
        "label": "Explain",
        "max_tokens": 512,
        "instruction": (
            "Answer style: give a fuller explanation — state the answer, then explain the "
            "relevant details and reasoning, drawn only from the context. Short paragraphs or "
            "bullet points are fine."
        ),
    },
}
DEFAULT_MODE = os.environ.get("RAG_MODE", "short").strip().lower()
if DEFAULT_MODE not in ANSWER_MODES:
    DEFAULT_MODE = "short"


def mode_config(mode: str | None) -> dict:
    """Return the config for `mode`, falling back to the default mode."""
    return ANSWER_MODES.get((mode or DEFAULT_MODE), ANSWER_MODES[DEFAULT_MODE])


def system_prompt(mode: str | None = None) -> str:
    """Build the system prompt for a given answer mode.

    The exact-refusal rule is stated last (recency) so it holds in both modes — the style
    directive only shapes how *grounded* answers read, never whether to refuse.
    """
    minfo = mode_config(mode)
    return (
        "You are an offline engineering assistant for CAN bus and embedded firmware topics.\n"
        "Answer ONLY using the provided context. Do not use outside knowledge.\n"
        f"{minfo['instruction']}\n"
        "When you use information, cite the source document name in square brackets, "
        "e.g. [can_fd_basics.md].\n"
        f'If the answer is not in the context, reply exactly: "{REFUSAL_TEXT}"'
    )


# Backwards-compatible default (short mode).
SYSTEM_PROMPT = system_prompt("short")

# --------------------------------------------------------------------------- #
# Conversation memory (multi-turn follow-ups).
# --------------------------------------------------------------------------- #
# How many prior (question, answer) turns to feed back into the prompt.
HISTORY_TURNS = int(os.environ.get("RAG_HISTORY_TURNS", "3"))
# Cap each remembered answer in the prompt so history can't blow the token budget.
HISTORY_ANSWER_CHARS = 400

# Stream answers token-by-token by default (feels faster on the slow iGPU). Disable with
# RAG_STREAM=0 or `--no-stream`; the non-streaming path is always available as a fallback.
STREAM_DEFAULT = os.environ.get("RAG_STREAM", "1").strip().lower() not in {"0", "false", "no"}


def context_char_budget() -> int:
    """Number of characters available for retrieved context inside the 4K budget."""
    usable_tokens = CONTEXT_TOKEN_LIMIT - MAX_ANSWER_TOKENS - PROMPT_OVERHEAD_TOKENS
    return max(usable_tokens, 0) * CHARS_PER_TOKEN
