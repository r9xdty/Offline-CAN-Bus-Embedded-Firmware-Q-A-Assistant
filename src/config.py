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
# Streamlit UI conversation history (multi-chat sidebar). Generated/local, gitignored.
CHATS_PATH = Path(os.environ.get("RAG_CHATS_PATH", DATA_DIR / "chats.json"))

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

# The prefix a general-knowledge answer (not grounded in the corpus) must start with, so the
# UI/pipeline can tell it apart from a grounded, cited answer at a glance.
GENERAL_LABEL = "[General knowledge — not from your documents]"


def is_general_answer(text: str) -> bool:
    """True if an answer is a labeled general-knowledge answer (tolerant of minor wording)."""
    return (text or "").lstrip().lower().startswith("[general knowledge")


# Master switch for the general-knowledge tier. On by default; set RAG_GENERAL_KNOWLEDGE=0 to
# restore the old strict grounded/refuse-only behavior.
GENERAL_KNOWLEDGE_ENABLED = os.environ.get("RAG_GENERAL_KNOWLEDGE", "1").strip().lower() not in {
    "0", "false", "no",
}

# Cosine floor above which a question is considered "in-domain enough" to let the model answer
# from general engineering knowledge when the corpus doesn't cover it. Deliberately set ABOVE
# MIN_SCORE: MIN_SCORE only decides whether a chunk is worth feeding to the model at all, while
# DOMAIN_SCORE gates a stronger claim -- "this question is clearly about CAN bus / embedded
# firmware" -- before we let the model speak without a citation. In practice (see README) an
# on-topic question's top match tends to land around ~0.5-0.7, while an off-topic one stays
# below ~0.1, so 0.25 sits comfortably in the gap. Override: RAG_DOMAIN_SCORE.
DOMAIN_SCORE = float(os.environ.get("RAG_DOMAIN_SCORE", "0.25"))

# Starter questions grounded in the sample corpus, for UI "try one of these" affordances.
EXAMPLE_QUESTIONS = [
    "What is the maximum bus length for CAN 2.0 at 500 kbps?",
    "How many data bytes can a CAN FD frame carry?",
    "What causes a node to go bus-off?",
    "How does the sample point relate to CAN bit timing on an STM32?",
    "What is a PGN in J1939?",
    "What is a PDO in CANopen?",
]

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


def system_prompt(mode: str | None = None, allow_general: bool = False) -> str:
    """Build the system prompt for a given answer mode.

    The exact-refusal rule is stated last (recency) so it holds in both modes — the style
    directive only shapes how *grounded* answers read, never whether to refuse.

    `allow_general` opens a third tier: when the context doesn't have the answer but the
    question is clearly on-topic, the model may answer from general engineering knowledge,
    labeled and uncited. It defaults to False so every existing caller (and `SYSTEM_PROMPT`
    below) keeps the original strict grounded/refuse-only prompt unchanged.
    """
    minfo = mode_config(mode)
    if not allow_general:
        return (
            "You are an offline engineering assistant for CAN bus and embedded firmware topics.\n"
            "Answer ONLY using the provided context. Do not use outside knowledge.\n"
            f"{minfo['instruction']}\n"
            "When you use information, cite the source document name in square brackets, "
            "e.g. [can_fd_basics.md].\n"
            f'If the answer is not in the context, reply exactly: "{REFUSAL_TEXT}"'
        )
    return (
        "You are an offline engineering assistant for CAN bus and embedded firmware topics.\n"
        "If the CONTEXT contains the answer, answer using ONLY the CONTEXT. Do not use outside "
        "knowledge and do not paraphrase away its exact terms, names, or numeric values -- "
        "preserve them as given.\n"
        f"{minfo['instruction']}\n"
        "When you use information from the context, cite the source document name in square "
        "brackets, e.g. [can_fd_basics.md].\n"
        "Only if the CONTEXT does not contain the answer, and the question is clearly about CAN "
        "bus or embedded firmware, you may answer from your own general engineering knowledge "
        f'instead. In that case, begin the answer with exactly "{GENERAL_LABEL}" and do not '
        "cite any source.\n"
        "Otherwise -- if you don't know, or the question is not about CAN bus or embedded "
        f'firmware -- reply exactly: "{REFUSAL_TEXT}"'
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
