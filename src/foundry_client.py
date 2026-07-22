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


def _norm(model_id: Any) -> str:
    """Normalize a model id for comparison: lowercase, drop any ':version' suffix."""
    return str(model_id).lower().split(":", 1)[0]


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


def _match(models: Any, target: str) -> Any:
    """Return a model or nested variant whose normalized id equals `target`, else None."""
    for m in models or []:
        if _norm(getattr(m, "id", "")) == target:
            return m
        for v in getattr(m, "variants", None) or []:
            if _norm(getattr(v, "id", "")) == target:
                return v
    return None


def _resolve_model(catalog: Any, model_id: str) -> Any:
    """Return the exact model variant for `model_id`, trying every catalog view.

    On-device GPU variants (OpenVINO / CUDA / TensorRT) are surfaced by ``get_cached_models``
    and each model's ``.variants`` — NOT by ``get_model_variant``/``list_models``, which only
    expose the remote ``*-generic-cpu`` catalog. So we search the cached models and variant
    lists too, and pin the exact variant so device placement stays deterministic.
    """
    target = _norm(model_id)

    # 1) Exact variant lookup against the remote catalog.
    try:
        model = catalog.get_model_variant(model_id)
        if model is not None:
            return model
    except Exception:
        pass

    # 2) Among the models actually cached on this machine (where GPU variants live).
    for source in (catalog.get_cached_models, catalog.list_models):
        try:
            found = _match(source(), target)
            if found is not None:
                return found
        except Exception:
            pass

    # 3) By alias, then match the requested variant among the model's variants.
    try:
        parent = catalog.get_model(_alias_of(model_id))
        if parent is not None:
            found = _match([parent], target)
            if found is not None:
                return found
    except Exception:
        pass

    # Failure: surface what the SDK can actually see so the fix is obvious.
    try:
        cached = [getattr(m, "id", "?") for m in catalog.get_cached_models()]
    except Exception:
        cached = []
    raise RuntimeError(
        f"Model '{model_id}' not found via the Foundry Local SDK.\n"
        f"Cached models the SDK reports: {cached or '(none)'}\n"
        "Confirm with `foundry model list --variants`; if the ID differs, update "
        "CHAT_MODEL_ID / EMBED_MODEL_ID in src/config.py."
    )


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
