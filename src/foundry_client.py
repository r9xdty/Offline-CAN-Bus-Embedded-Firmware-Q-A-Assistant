"""Foundry Local integration: chat client (Intel iGPU) + embedding client (NVIDIA GPU).

This is the only module that talks to Foundry Local. Everything else depends on the two
callables it exposes:

    embed_texts(texts) -> np.ndarray   # raw (un-normalized) float32 vectors, shape (n, dim)
    chat(messages)     -> str          # grounded answer text

The Foundry Local runtime and its models are only available on the target machine (Windows +
Intel iGPU + NVIDIA RTX 3050 Ti). The SDK is imported lazily inside `get_clients()` so that
the rest of the package — chunking, cosine search, SQLite storage, prompt building — can be
imported and unit-tested on any machine without the runtime present.

API notes (foundry-local-sdk / foundry-local-sdk-winml, v1.x):

    from foundry_local_sdk import Configuration, FoundryLocalManager
    from foundry_local_sdk.openai import ChatClientSettings

    FoundryLocalManager.initialize(Configuration(app_name="rag-can-assistant"))
    catalog = FoundryLocalManager.instance.catalog
    model = catalog.get_model_variant("phi-4-mini-instruct-openvino-gpu")  # exact variant
    model.download(); model.load()
    chat = model.get_chat_client()
    chat.settings = ChatClientSettings(temperature=0.1, max_tokens=512)
    resp = chat.complete_chat(messages)          # OpenAI-shaped ChatCompletion
    text = resp.choices[0].message.content

    emb = catalog.get_model_variant("qwen3-embedding-0.6b-cuda-gpu")
    emb.download(); emb.load()
    r = emb.get_embedding_client().generate_embeddings(["..."])  # CreateEmbeddingResponse
    vector = r.data[0].embedding                 # list[float]

Models are initialized once and kept loaded for the process lifetime so each query is fast.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from . import config


@dataclass
class _Clients:
    manager: Any
    chat_model: Any
    embed_model: Any
    chat_client: Any
    embed_client: Any


_clients: _Clients | None = None
_lock = threading.Lock()


def _resolve_model(catalog: Any, model_id: str) -> Any:
    """Return the exact model variant for `model_id`, falling back to alias lookup.

    Pinning the full variant ID keeps device placement deterministic; if the catalog has
    shifted and the exact variant is missing, fall back to the alias (its leading segment).
    """
    model = catalog.get_model_variant(model_id)
    if model is None:
        # Best-effort alias: strip the trailing device/EP qualifier tokens.
        alias = model_id
        for suffix in (
            "-openvino-gpu",
            "-cuda-gpu",
            "-generic-cpu",
            "-trtrtx-gpu",
            "-generic-gpu",
        ):
            if alias.endswith(suffix):
                alias = alias[: -len(suffix)]
                break
        model = catalog.get_model(alias)
    if model is None:
        raise RuntimeError(
            f"Model '{model_id}' not found in the Foundry Local catalog. "
            "Run `foundry model list --variants` to see available/cached models."
        )
    return model


def _prepare(model: Any) -> None:
    """Download (if needed) and load a model so it is resident and ready to serve."""
    if not getattr(model, "is_cached", False):
        model.download()
    if not getattr(model, "is_loaded", False):
        model.load()


def get_clients() -> _Clients:
    """Initialize Foundry Local once and return cached chat + embedding clients.

    Idempotent and thread-safe. The chat model lands on the Intel iGPU (OpenVINO) and the
    embedding model on the NVIDIA GPU (CUDA) purely by virtue of the pinned variant IDs.
    """
    global _clients
    if _clients is not None:
        return _clients

    with _lock:
        if _clients is not None:
            return _clients

        # Lazy import: only needed when we actually talk to the runtime.
        from foundry_local_sdk import Configuration, FoundryLocalManager
        from foundry_local_sdk.openai import ChatClientSettings

        if FoundryLocalManager.instance is None:
            FoundryLocalManager.initialize(Configuration(app_name=config.APP_NAME))
        manager = FoundryLocalManager.instance
        catalog = manager.catalog

        chat_model = _resolve_model(catalog, config.CHAT_MODEL_ID)
        embed_model = _resolve_model(catalog, config.EMBED_MODEL_ID)
        _prepare(chat_model)
        _prepare(embed_model)

        chat_client = chat_model.get_chat_client()
        chat_client.settings = ChatClientSettings(
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_ANSWER_TOKENS,
        )
        embed_client = embed_model.get_embedding_client()

        _clients = _Clients(
            manager=manager,
            chat_model=chat_model,
            embed_model=embed_model,
            chat_client=chat_client,
            embed_client=embed_client,
        )
        return _clients


def warmup() -> None:
    """Pre-load both models up front (e.g. at CLI/Streamlit startup)."""
    get_clients()


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts on the NVIDIA GPU. Returns raw float32 array (n, dim).

    Vectors are returned un-normalized; callers normalize with `vectors.l2_normalize`.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    resp = get_clients().embed_client.generate_embeddings(list(texts))
    # OpenAI-shaped response: data items carry `.index` and `.embedding`.
    ordered = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
    return np.asarray([d.embedding for d in ordered], dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns a 1-D raw float32 vector."""
    resp = get_clients().embed_client.generate_embedding(text)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def chat(messages: List[dict]) -> str:
    """Run a grounded chat completion on the Intel iGPU and return the answer text."""
    resp = get_clients().chat_client.complete_chat(messages)
    return (resp.choices[0].message.content or "").strip()
