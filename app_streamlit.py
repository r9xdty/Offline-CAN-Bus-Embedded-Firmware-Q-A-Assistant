"""Streamlit UI — Phase 2 interface + document upload (spec §10.2).

Run with::

    streamlit run app_streamlit.py

Left sidebar: manage the knowledge base — upload embedded-systems PDFs / Markdown / text files,
see what's indexed, and remove documents. Main pane: ask a question and get a grounded answer
with sources and a retrieved-chunk expander.

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


def _sidebar() -> None:
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

    st.sidebar.divider()
    sources = ingest.list_sources()
    if not sources:
        st.sidebar.caption("Empty. Upload files above, or run `python -m src.ingest`.")
        return

    total = sum(n for _, n in sources)
    st.sidebar.caption(f"{len(sources)} documents · {total} chunks")
    for source, n in sources:
        col_name, col_btn = st.sidebar.columns([0.78, 0.22])
        col_name.write(f"`{source}`  \n{n} chunks")
        if col_btn.button("✕", key=f"rm_{source}", help=f"Remove {source}"):
            ingest.remove_source(source)
            _refresh_pipeline()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="CAN Bus / Firmware Q&A", page_icon="🔧", layout="wide")
    st.title("🔧 Offline CAN Bus / Embedded Firmware Q&A")
    st.caption(
        "Offline retrieval-augmented Q&A over your local embedded-systems corpus. "
        "Chat on the Intel iGPU (OpenVINO), embeddings on the NVIDIA GPU (CUDA)."
    )

    _sidebar()

    pipeline = get_pipeline()
    if pipeline.size == 0:
        st.info(
            "The knowledge base is empty. Upload PDF / Markdown / text files from the sidebar, "
            "or run `python -m src.ingest` to build the starter CAN-bus corpus."
        )
        return

    st.markdown(f"**{pipeline.size} chunks indexed.** Ask a CAN-bus or firmware question.")

    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What is the maximum CAN bus length at 500 kbps?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        with st.spinner("Retrieving and generating..."):
            result = pipeline.answer(question)

        st.subheader("Answer")
        st.write(result.answer)

        if result.sources:
            st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in result.sources))
        else:
            st.markdown("**Sources:** _(none — answer not grounded in the corpus)_")

        with st.expander("Retrieved chunks & similarity scores"):
            if not result.chunks:
                st.write("No chunks retrieved.")
            for ch in result.chunks:
                st.markdown(f"**`{ch.source}` #{ch.chunk_index}** — score `{ch.score:.4f}`")
                st.text(ch.content)
                st.divider()
    elif submitted:
        st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
