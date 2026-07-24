"""Streamlit UI — Phase 2 interface (spec §10.2): chat with memory, modes, and document upload.

Run with::

    streamlit run app_streamlit.py

Left sidebar: a clear-conversation button, the conversation switcher, and the knowledge-base
manager (upload embedded-systems PDFs / Markdown / text, see what's indexed, remove documents).
Main pane: an answer-mode selector (short / explain) and a chat that remembers the conversation,
so follow-ups like "explain that" work.

The Foundry client and knowledge base are cached so each query is fast. Uploading requires the
Foundry Local server to be running (it embeds the new chunks on the NVIDIA GPU).
"""

from __future__ import annotations

import streamlit as st

from src import config, conversations, documents, ingest, smalltalk
from src.pipeline import Pipeline


def _inject_css() -> None:
    """Inject presentation-only CSS for a clean, modern chat look (light + dark, warm accent).

    Defensive by design: every rule targets a `data-testid`/class Streamlit has shipped for a
    long time, and nothing here touches app behavior — if a selector doesn't match on some
    Streamlit version, the affected element just keeps its default look and the app still works.
    No external resources (fonts, images, CSS) are loaded — the app is fully offline.
    """
    st.markdown(
        """
        <style>
        /* ---------------------------------------------------------------- */
        /* System font stack only — no external/@import fonts (offline app). */
        /* ---------------------------------------------------------------- */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
                Arial, sans-serif;
        }

        /* ---------------------------------------------------------------- */
        /* Hide Streamlit chrome we don't need, keep the sidebar toggle.     */
        /* ---------------------------------------------------------------- */
        #MainMenu { visibility: hidden; }
        [data-testid="stToolbarActions"] { visibility: hidden; }
        footer { visibility: hidden; }

        /* ---------------------------------------------------------------- */
        /* Comfortable, centered reading column (layout stays "wide" so the */
        /* sidebar keeps its own room).                                     */
        /* ---------------------------------------------------------------- */
        .block-container {
            max-width: 820px;
            padding-top: 2rem;
        }

        /* ---------------------------------------------------------------- */
        /* Chat bubbles: rounded, roomy, theme-agnostic translucent fill so */
        /* it reads correctly on both light and dark backgrounds.           */
        /* ---------------------------------------------------------------- */
        [data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.9rem;
            background-color: rgba(127, 127, 127, 0.08);
            line-height: 1.55;
        }

        /* ---------------------------------------------------------------- */
        /* Buttons + example-question chips: softer, pill/card-like.        */
        /* ---------------------------------------------------------------- */
        .stButton > button {
            border-radius: 10px;
        }

        /* ---------------------------------------------------------------- */
        /* Sidebar: tidy header spacing, full-width left-aligned chat rows  */
        /* with a hover background; the active chat (type="primary") gets  */
        /* a subtle extra emphasis on top of the theme's accent color.      */
        /* ---------------------------------------------------------------- */
        [data-testid="stSidebar"] h2 {
            margin-top: 0.6rem;
        }
        [data-testid="stSidebar"] .stButton > button {
            text-align: left;
            justify-content: flex-start;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background-color: rgba(127, 127, 127, 0.12);
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            font-weight: 600;
        }

        /* ---------------------------------------------------------------- */
        /* Dark-mode tweaks — kept minimal since translucent backgrounds     */
        /* already adapt to both themes automatically.                      */
        /* ---------------------------------------------------------------- */
        @media (prefers-color-scheme: dark) {
            [data-testid="stChatMessage"] {
                background-color: rgba(255, 255, 255, 0.06);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _render_conversations_sidebar() -> None:
    """Render the "Conversations" section: new-chat button, switcher, rename, and delete.

    Reads/writes `st.session_state.chats` (loaded by `main()` before `_sidebar()` runs) and
    persists every mutation to disk so chat history survives an app restart.
    """
    st.sidebar.header("Conversations")
    chats = st.session_state.chats

    if st.sidebar.button("➕ New chat", use_container_width=True):
        conversations.new_conversation(chats)
        conversations.save(config.CHATS_PATH, chats)
        st.rerun()

    convo_list = chats["conversations"]
    current_id = chats.get("current_id")
    editing_id = st.session_state.get("editing_chat")

    # One row per conversation: a full-width switch button (standing in for the old radio
    # switcher — active chat renders as primary/accent-colored, the rest as secondary) plus a
    # small edit affordance that swaps the row into an inline rename text_input.
    for conv in convo_list:
        cid = conv["id"]
        title = conv.get("title") or conversations.DEFAULT_TITLE
        col_main, col_action = st.sidebar.columns([0.82, 0.18])
        if cid == editing_id:
            col_main.text_input(
                "Chat name", value=title, key=f"edit_{cid}", label_visibility="collapsed"
            )
            if col_action.button("✓", key=f"save_{cid}"):
                new_title = st.session_state.get(f"edit_{cid}", title)
                conversations.rename_conversation(chats, cid, new_title)
                st.session_state.editing_chat = None
                conversations.save(config.CHATS_PATH, chats)
                st.rerun()
        else:
            label = title if len(title) <= 28 else title[:27].rstrip() + "…"
            is_active = cid == current_id
            if col_main.button(
                label,
                key=f"switch_{cid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ) and not is_active:
                chats["current_id"] = cid
                conversations.save(config.CHATS_PATH, chats)
                st.rerun()
            if col_action.button("✎", key=f"edit_btn_{cid}"):
                st.session_state.editing_chat = cid
                st.rerun()

    if st.sidebar.button("🗑 Delete current chat", use_container_width=True):
        conversations.delete_conversation(chats, current_id)
        conversations.ensure_current(chats)
        conversations.save(config.CHATS_PATH, chats)
        st.rerun()


def _sidebar() -> bool:
    """Render sidebar controls; return the selected general-knowledge toggle.

    (Answer mode now lives in the main pane — see `main()`.)
    """
    st.sidebar.header("Chat")
    general_on = st.sidebar.toggle(
        "General-knowledge fallback",
        value=config.GENERAL_KNOWLEDGE_ENABLED,
        help="When on, on-topic questions not covered by your corpus get a labeled "
             "general-knowledge answer; when off, they are refused.",
    )
    if st.sidebar.button("🧹 Clear conversation"):
        conversations.current(st.session_state.chats)["messages"].clear()
        conversations.save(config.CHATS_PATH, st.session_state.chats)
        st.rerun()

    st.sidebar.divider()
    _render_conversations_sidebar()

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
        col_docs, col_chunks = st.sidebar.columns(2)
        col_docs.metric("Documents", len(sources))
        col_chunks.metric("Chunks", total)
        for source, n in sources:
            col_name, col_btn = st.sidebar.columns([0.78, 0.22])
            col_name.write(f"`{source}`  \n{n} chunks")
            if col_btn.button("✕", key=f"rm_{source}", help=f"Remove {source}"):
                ingest.remove_source(source)
                _refresh_pipeline()
                st.rerun()

    with st.sidebar.expander("How it works"):
        st.markdown(
            "- Answers are retrieved from your local corpus and cited by source document.\n"
            "- If a question is on-topic but not covered, the assistant may fall back to "
            "general engineering knowledge — always clearly labeled, never cited.\n"
            "- Off-topic questions are refused rather than guessed at.\n"
            "- Everything runs offline: chat on the Intel iGPU, embeddings on the NVIDIA GPU."
        )
    return general_on


def _render_turn(turn: dict) -> None:
    """Render one saved (question, answer) exchange as chat messages."""
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(turn["question"])
    with st.chat_message("assistant", avatar="🔧"):
        st.write(turn["answer"])
        if turn.get("smalltalk"):
            return  # greeting/meta reply: no sources, score, or chunk expander
        kind = turn.get("kind", "grounded")
        if kind == "general":
            st.warning(
                "General knowledge — not grounded in your documents. This comes from the "
                "model's general engineering knowledge, not your uploaded corpus."
            )
        elif turn["sources"]:
            st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in turn["sources"]))
        else:
            st.markdown("**Sources:** _(none — not grounded in the corpus)_")
        top = f"{turn['top_score']:.2f}" if turn["top_score"] is not None else "n/a"
        st.caption(f"{turn['mode']} · answered in {turn['elapsed_s']:.1f}s · top match {top}")
        if turn["chunks"]:
            with st.expander("Retrieved chunks & similarity scores"):
                for src, idx, score, content in turn["chunks"]:
                    st.markdown(f"**`{src}` #{idx}** — score `{score:.4f}`")
                    st.progress(min(max(score, 0.0), 1.0))
                    st.text(content)
                    st.divider()


def main() -> None:
    st.set_page_config(page_title="CAN Bus / Firmware Q&A", page_icon="🔧", layout="wide")
    _inject_css()

    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.1rem;'>"
        "<span style='font-size:2rem;line-height:1;'>🔧</span>"
        "<span style='font-size:1.6rem;font-weight:700;'>Offline CAN Bus / Embedded Firmware Q&A</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Offline, grounded & cited — with labeled general-knowledge fallback and refusal "
        "for off-topic questions."
    )

    if "chats" not in st.session_state:
        st.session_state.chats = conversations.load(config.CHATS_PATH)
    conversations.ensure_current(st.session_state.chats)

    general_on = _sidebar()

    # The active conversation's message list. It's the SAME list object stored inside
    # st.session_state.chats, so appending to `messages` mutates the store in place —
    # we still call conversations.save() after each mutation to persist it to disk.
    messages = conversations.current(st.session_state.chats)["messages"]

    pipeline = get_pipeline()
    if pipeline.size == 0:
        st.info(
            "The knowledge base is empty. Upload PDF / Markdown / text files from the sidebar, "
            "or run `python -m src.ingest` to build the starter CAN-bus corpus."
        )
        return

    mode = st.radio(
        "Answer mode",
        options=list(config.ANSWER_MODES),
        format_func=lambda m: config.ANSWER_MODES[m]["label"],
        index=list(config.ANSWER_MODES).index(config.DEFAULT_MODE),
        horizontal=True,
        help="Short = a direct 1-2 sentence answer. Explain = a fuller, explained answer.",
        key="mode_select",
    )
    st.caption(f"{pipeline.size} chunks indexed")

    if not messages:
        st.markdown("#### 👋 Ask me about CAN bus & embedded firmware")
        st.caption("Try one of these to get started:")
        cols = st.columns(2)
        for i, example in enumerate(config.EXAMPLE_QUESTIONS):
            if cols[i % 2].button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

    for turn in messages:
        _render_turn(turn)

    question = st.session_state.pop("pending_question", None) or st.chat_input(
        "Ask a CAN-bus or firmware question (follow-ups are remembered)…"
    )
    if question and question.strip():
        chit_chat = smalltalk.reply(question)
        if chit_chat is not None:
            # Not a grounded turn: show it, but flag it so it's excluded from pipeline history.
            messages.append({"question": question, "answer": chit_chat, "smalltalk": True})
            conversations.save(config.CHATS_PATH, st.session_state.chats)
            st.rerun()

        # Memory excludes small-talk turns so a greeting can't pollute follow-up retrieval.
        history = [
            (m["question"], m["answer"])
            for m in messages
            if not m.get("smalltalk")
        ]
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(question)
        with st.chat_message("assistant", avatar="🔧"):
            placeholder = st.empty()
            placeholder.markdown("▌")
            acc = {"text": ""}

            def _on_token(tok: str) -> None:
                acc["text"] += tok
                placeholder.markdown(acc["text"] + "▌")

            stream = config.STREAM_DEFAULT
            result = pipeline.answer(
                question, history=history, mode=mode,
                on_token=_on_token if stream else None,
                general_enabled=general_on,
            )
            placeholder.markdown(result.answer)  # final text without the cursor
        messages.append(
            {
                "question": result.question,
                "answer": result.answer,
                "sources": result.sources,
                "top_score": result.top_score,
                "elapsed_s": result.elapsed_s,
                "mode": result.mode,
                "chunks": [(c.source, c.chunk_index, c.score, c.content) for c in result.chunks],
                "kind": result.kind,
            }
        )
        conversations.touch_title(conversations.current(st.session_state.chats))
        conversations.save(config.CHATS_PATH, st.session_state.chats)
        st.rerun()


if __name__ == "__main__":
    main()
