"""Foundry Local integration via the running server's OpenAI-compatible HTTP endpoint.

This is the only module that talks to Foundry Local. Everything else depends on the two
callables it exposes:

    embed_texts(texts) -> np.ndarray   # raw (un-normalized) float32 vectors, shape (n, dim)
    chat(messages)     -> str          # grounded answer text

**Why HTTP and not the in-process SDK core.** The `foundry-local-sdk` native core runs
in-process and, in practice, does not share the running `foundry server` daemon's model cache
or its downloaded execution-provider packs (OpenVINO / CUDA / TensorRT). On a real machine it
reports zero cached models and only the remote `*-generic-cpu` catalog, so the pinned GPU
variants can't be resolved through it. The daemon, however, already has everything working and
exposes an OpenAI-compatible endpoint. Talking to that endpoint over localhost HTTP also
sidesteps Windows admin/normal-user cache-visibility issues. The build spec explicitly allows
using the `openai` client against the Foundry endpoint.

Prerequisite: the server must be running (`foundry server start`). Models are loaded into the
server on demand — the server lists cached models but does not auto-load them, so a first call
triggers `foundry model load <id>` (idempotent) and then retries.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from typing import Any, List

import numpy as np

from . import config

_client: Any = None
_endpoint: str | None = None
_loaded: set[str] = set()
_lock = threading.Lock()


def _alias_of(model_id: str) -> str:
    """Derive the catalog alias (e.g. 'phi-4-mini') from a full variant id."""
    alias = model_id
    for suffix in ("-openvino-gpu", "-cuda-gpu", "-trtrtx-gpu", "-generic-gpu", "-generic-cpu"):
        if alias.endswith(suffix):
            alias = alias[: -len(suffix)]
            break
    if alias.endswith("-instruct"):
        alias = alias[: -len("-instruct")]
    return alias


def _discover_endpoint() -> str:
    """Return the base URL of the running Foundry Local server.

    Order: the FOUNDRY_LOCAL_ENDPOINT env var / config value, else parse the URL out of
    `foundry server status`. Raises a clear error if neither yields an endpoint.
    """
    ep = config.FOUNDRY_ENDPOINT
    if ep:
        return ep.rstrip("/")

    exe = shutil.which("foundry")
    if exe:
        try:
            out = subprocess.run(
                [exe, "server", "status"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # foundry output isn't the console code page (cp1254 etc.)
                timeout=20,
            ).stdout
            match = re.search(r"https?://[\w.]+:\d+", out or "")
            if match:
                return match.group(0).rstrip("/")
        except Exception:
            pass

    raise RuntimeError(
        "Could not find the Foundry Local server endpoint.\n"
        "Start it with `foundry server start`, then either run from a shell where the "
        "`foundry` CLI is on PATH (this reads `foundry server status`) or set the "
        "FOUNDRY_LOCAL_ENDPOINT env var to the Web URL it prints, e.g.\n"
        "    setx FOUNDRY_LOCAL_ENDPOINT http://127.0.0.1:54163"
    )


def _base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1") else endpoint + "/v1"


def get_client() -> Any:
    """Return a cached OpenAI client pointed at the running Foundry server. Thread-safe."""
    global _client, _endpoint
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        from openai import OpenAI  # lazy: only needed when we actually talk to the server

        _endpoint = _discover_endpoint()
        # The local server ignores the API key, but the OpenAI client requires a non-empty one.
        # A bounded timeout means a stuck request errors out instead of freezing forever.
        _client = OpenAI(
            base_url=_base_url(_endpoint),
            api_key="foundry-local",
            timeout=config.REQUEST_TIMEOUT,
            max_retries=0,
        )
        return _client


def _ensure_loaded(model_id: str) -> None:
    """Best-effort: load `model_id` into the running server via the CLI (idempotent).

    Tries the full variant id first, then the alias. Failures are swallowed here; if the model
    still isn't loaded the subsequent request raises a clear error via `_call`.
    """
    if model_id in _loaded:
        return
    exe = shutil.which("foundry")
    if exe:
        for target in (model_id, _alias_of(model_id)):
            try:
                # Discard output as raw bytes: we only need the exit code, and the CLI's
                # progress bars aren't the console code page (cp1254 etc.), so decoding them
                # in subprocess reader threads would raise UnicodeDecodeError.
                result = subprocess.run(
                    [exe, "model", "load", target],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=600,
                )
                if result.returncode == 0:
                    break
            except Exception:
                pass
    _loaded.add(model_id)


def _is_not_loaded(exc: Exception) -> bool:
    return "not loaded" in str(exc).lower()


def _is_timeout(exc: Exception) -> bool:
    return "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()


def _timeout_error(model_id: str) -> RuntimeError:
    return RuntimeError(
        f"The Foundry server did not respond within {config.REQUEST_TIMEOUT:.0f}s for model "
        f"'{model_id}'. On the Intel iGPU (OpenVINO) the first load compiles the model and can "
        f"be very slow. Either wait longer (raise RAG_REQUEST_TIMEOUT) or switch to the fast "
        f"cached NVIDIA/TensorRT model:  setx RAG_CHAT_MODEL phi-3.5-mini-instruct-trtrtx-gpu"
    )


def _call(fn, model_id: str):
    """Run `fn()`; on 'model not loaded' load the model once and retry; map timeouts to a hint."""
    try:
        return fn()
    except Exception as exc:
        if _is_timeout(exc):
            raise _timeout_error(model_id) from exc
        if not _is_not_loaded(exc):
            raise
        _ensure_loaded(model_id)
        try:
            return fn()
        except Exception as exc2:
            if _is_timeout(exc2):
                raise _timeout_error(model_id) from exc2
            if _is_not_loaded(exc2):
                raise RuntimeError(
                    f"Foundry Local could not load model '{model_id}'. "
                    f"Load it manually with `foundry model load {model_id}` "
                    f"(or `foundry model load {_alias_of(model_id)}`) and retry."
                ) from exc2
            raise


def warmup() -> None:
    """Pre-load both models into the server up front (optional; speeds the first query)."""
    get_client()
    _ensure_loaded(config.EMBED_MODEL_ID)
    _ensure_loaded(config.CHAT_MODEL_ID)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts on the NVIDIA GPU. Returns raw float32 array (n, dim).

    Vectors are returned un-normalized; callers normalize with `vectors.l2_normalize`.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    client = get_client()
    resp = _call(
        lambda: client.embeddings.create(model=config.EMBED_MODEL_ID, input=list(texts)),
        config.EMBED_MODEL_ID,
    )
    ordered = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
    return np.asarray([d.embedding for d in ordered], dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns a 1-D raw float32 vector."""
    client = get_client()
    resp = _call(
        lambda: client.embeddings.create(model=config.EMBED_MODEL_ID, input=[text]),
        config.EMBED_MODEL_ID,
    )
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def chat(messages: List[dict], max_tokens: int | None = None) -> str:
    """Run a grounded chat completion on the Intel iGPU and return the answer text."""
    client = get_client()
    resp = _call(
        lambda: client.chat.completions.create(
            model=config.CHAT_MODEL_ID,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=max_tokens or config.MAX_ANSWER_TOKENS,
        ),
        config.CHAT_MODEL_ID,
    )
    return (resp.choices[0].message.content or "").strip()
