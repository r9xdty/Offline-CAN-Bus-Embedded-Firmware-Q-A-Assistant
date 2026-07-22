"""Streamlit UI — Phase 2 interface (spec §10.2).

Run with::

    streamlit run app_streamlit.py

A text input, a submit button, the grounded answer, and an expander showing the retrieved
chunks + sources. The Foundry client and knowledge base are loaded once (cached) so each query
is fast.
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.pipeline import Pipeline


@st.cache_resource(show_spinner="Loading models and knowledge base (first run may download models)...")
def get_pipeline() -> Pipeline:
    """Build the pipeline once per Streamlit process (models + KB stay warm)."""
    return Pipeline()


def main() -> None:
    st.set_page_config(page_title="CAN Bus / Firmware Q&A", page_icon="🔧")
    st.title("🔧 Offline CAN Bus / Firmware Q&A")
    st.caption(
        "Fully offline retrieval-augmented Q&A over a local CAN-bus / embedded-firmware corpus. "
        "Chat on the Intel iGPU (OpenVINO), embeddings on the NVIDIA GPU (CUDA)."
    )

    pipeline = get_pipeline()

    if pipeline.size == 0:
        st.error(
            "The knowledge base is empty. Populate `data/raw/` and run "
            "`python -m src.ingest`, then reload this page."
        )
        return

    st.info(f"{pipeline.size} chunks indexed. Ask a CAN-bus or firmware question.")

    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What is the maximum CAN bus length at 500 kbps?",
        )
        submitted = st.form_submit_button("Ask")

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
                st.markdown(
                    f"**`{ch.source}` #{ch.chunk_index}** — score `{ch.score:.4f}`"
                )
                st.text(ch.content)
                st.divider()
    elif submitted:
        st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
