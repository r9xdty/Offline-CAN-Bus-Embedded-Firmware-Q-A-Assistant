"""Streamlit UI — Phase 2 interface (spec §10.2): chat with memory, modes, and document upload.

Run with::

    streamlit run app_streamlit.py

Left sidebar: answer mode (short / explain), a clear-conversation button, and the knowledge-base
manager (upload embedded-systems PDFs / Markdown / text, see what's indexed, remove documents).
Main pane: a chat that remembers the conversation, so follow-ups like "explain that" work.

The Foundry client and knowledge base are cached so each query is fast. Uploading requires the
Foundry Local server to be running (it embeds the new chunks on the NVIDIA GPU).
"""

from __future__ import annotations

import streamlit as st

from src import config, documents, ingest
from src.pipeline import Pipeline


@st.cache_resource(show_spinner="Loading models and knowledge base...")
def get_pipeline() -> Pipeline:
    """Build the pipeline once per Streamlit process (models + KB stay warm)."""
    return Pipeline()


def _refresh_pipeline() -> None:
    """Drop the cached pipeline so the next run reloads the KB after an upload/removal."""
    get_pipeline.clear()


def _handle_uploads(files) -> None:
    """Extract text from each uploaded file and upsert it into the knowledge base."""
    added, skipped, failed = [], [], []
    progress = st.progress(0.0)
    for i, up in enumerate(files, start=1):
        try:
            text = documents.extract_text(up.name, up.getvalue())
            if not text.strip():
                skipped.append(up.name)  # e.g. a scanned/image-only PDF with no text layer
            else:
                n = ingest.add_document(up.name, text)
                added.append((up.name, n))
        except Exception as exc:  # noqa: BLE001 - surface any extraction/embedding error
            failed.append((up.name, str(exc)))
        progress.progress(i / len(files))
    progress.empty()

    for name, n in added:
        st.success(f"Added **{name}** ({n} chunks).")
    for name in skipped:
        st.warning(f"No extractable text in **{name}** — is it a scanned/image-only PDF? "
                   "(OCR is out of scope.)")
    for name, err in failed:
        st.error(f"Failed to add **{name}**: {err}")

    if added:
        _refresh_pipeline()


def _sidebar() -> str:
    """Render sidebar controls; return the selected answer mode."""
    st.sidebar.header("Chat")
    mode = st.sidebar.radio(
        "Answer mode",
        options=list(config.ANSWER_MODES),
        format_func=lambda m: config.ANSWER_MODES[m]["label"],
        index=list(config.ANSWER_MODES).index(config.DEFAULT_MODE),
        help="Short = a direct 1-2 sentence answer. Explain = a fuller, explained answer.",
    )
    if st.sidebar.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Knowledge base")
    files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "md", "markdown", "txt"],
        accept_multiple_files=True,
        help="Embedded-systems datasheets, app notes, reference-manual chapters. "
             "PDFs are converted to text automatically.",
    )
    if files and st.sidebar.button("Add to knowledge base", type="primary"):
        _handle_uploads(files)

    sources = ingest.list_sources()
    if not sources:
        st.sidebar.caption("Empty. Upload files above, or run `python -m src.ingest`.")
    else:
        total = sum(n for _, n in sources)
        st.sidebar.caption(f"{len(sources)} documents · {total} chunks")
        for source, n in sources:
            col_name, col_btn = st.sidebar.columns([0.78, 0.22])
            col_name.write(f"`{source}`  \n{n} chunks")
            if col_btn.button("✕", key=f"rm_{source}", help=f"Remove {source}"):
                ingest.remove_source(source)
                _refresh_pipeline()
                st.rerun()
    return mode


def _render_turn(turn: dict) -> None:
    """Render one saved (question, answer) exchange as chat messages."""
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in turn["sources"]))
        else:
            st.markdown("**Sources:** _(none — not grounded in the corpus)_")
        top = f"{turn['top_score']:.2f}" if turn["top_score"] is not None else "n/a"
        st.caption(f"{turn['mode']} · answered in {turn['elapsed_s']:.1f}s · top match {top}")
        if turn["chunks"]:
            with st.expander("Retrieved chunks & similarity scores"):
                for src, idx, score, content in turn["chunks"]:
                    st.markdown(f"**`{src}` #{idx}** — score `{score:.4f}`")
                    st.text(content)
                    st.divider()


def main() -> None:
    st.set_page_config(page_title="CAN Bus / Firmware Q&A", page_icon="🔧", layout="wide")
    st.title("🔧 Offline CAN Bus / Embedded Firmware Q&A")
    st.caption(
        "Offline retrieval-augmented Q&A over your local embedded-systems corpus, with memory. "
        "Chat on the Intel iGPU (OpenVINO), embeddings on the NVIDIA GPU (CUDA)."
    )

    mode = _sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    pipeline = get_pipeline()
    if pipeline.size == 0:
        st.info(
            "The knowledge base is empty. Upload PDF / Markdown / text files from the sidebar, "
            "or run `python -m src.ingest` to build the starter CAN-bus corpus."
        )
        return

    st.caption(f"{pipeline.size} chunks indexed · mode: **{config.ANSWER_MODES[mode]['label']}**")

    for turn in st.session_state.messages:
        _render_turn(turn)

    question = st.chat_input("Ask a CAN-bus or firmware question (follow-ups are remembered)…")
    if question and question.strip():
        history = [(m["question"], m["answer"]) for m in st.session_state.messages]
        with st.spinner("Retrieving and generating…"):
            result = pipeline.answer(question, history=history, mode=mode)
        st.session_state.messages.append(
            {
                "question": result.question,
                "answer": result.answer,
                "sources": result.sources,
                "top_score": result.top_score,
                "elapsed_s": result.elapsed_s,
                "mode": result.mode,
                "chunks": [(c.source, c.chunk_index, c.score, c.content) for c in result.chunks],
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
