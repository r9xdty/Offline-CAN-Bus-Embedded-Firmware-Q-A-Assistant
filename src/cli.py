"""CLI — Phase 1 interface (spec §10.1), with conversation memory and answer modes.

    python -m src.cli                    # ask questions in a loop (remembers the conversation)
    python -m src.cli --debug            # also print retrieved chunks + similarity scores
    python -m src.cli --mode explain     # start in "explain" mode (default: short)

In-session commands:
    :short / :explain    switch answer mode
    :reset (:clear)      forget the conversation so far
    :help                show commands

Type `quit` (or `q`, `bye`, an empty line, Ctrl+C) to exit.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import List, Tuple

from . import config, foundry_client, smalltalk
from .pipeline import Answer, Pipeline

QUIT_WORDS = {
    "quit",
    "exit",
    "bye",
    "q",
    "hadi sg",
    "thank you for your services",
    "thank you for your soul",
}

GOODBYE_MESSAGES = ["Bye", "hadi sende sg", "Adamsin Quershma", "Gorusuruz knk", "babays"]

_HELP = (
    "Commands:\n"
    "  :short / :explain   switch answer mode (short = 1-2 sentences, explain = fuller)\n"
    "  :reset (:clear)     forget the conversation so far\n"
    "  :help               show this help\n"
    "  quit / q / bye      exit"
)


def _print_meta(result: Answer, debug: bool) -> None:
    """Print the sources + stats line (and, with --debug, the retrieved chunks)."""
    if result.sources:
        print(f"Sources: {result.sources}")
    else:
        print("Sources: []")
    top = f"{result.top_score:.2f}" if result.top_score is not None else "n/a"
    print(f"({result.elapsed_s:.1f}s · {result.mode} · top match {top})")
    if debug:
        print("\n--- retrieved chunks ---")
        if not result.chunks:
            print("  (none)")
        for ch in result.chunks:
            preview = ch.content.replace("\n", " ")
            print(f"  score={ch.score:.4f}  [{ch.source}#{ch.chunk_index}]  {preview[:140]}...")
    print()


def _handle_command(cmd: str, mode: str, history: List[Tuple[str, str]]) -> str:
    """Apply an in-session `:command`. Returns the (possibly changed) mode."""
    if cmd in config.ANSWER_MODES:
        print(f"[mode: {cmd}]")
        return cmd
    if cmd in {"reset", "clear", "new"}:
        history.clear()
        print("[conversation cleared]")
    elif cmd in {"help", "?"}:
        print(_HELP)
    else:
        print(f"[unknown command ':{cmd}' — try :help]")
    return mode


def run(debug: bool = False, mode: str | None = None, stream: bool = True) -> None:
    mode = mode or config.DEFAULT_MODE
    print("Loading models and knowledge base (first run may download models)...")
    pipeline = Pipeline()
    if pipeline.size == 0:
        print(
            "Knowledge base is empty. Populate data/raw/ and run `python -m src.ingest` first.",
            file=sys.stderr,
        )
        return

    # Pre-load the models into the server up front so the slow first-load (the iGPU/OpenVINO
    # compile can take a while) happens here with a message, not as a mid-question freeze.
    print("Warming up models (first load can take a while on the iGPU)...")
    try:
        foundry_client.warmup()
    except Exception as exc:  # noqa: BLE001 - non-fatal; the first query will surface details
        print(f"  (warm-up skipped: {exc})", file=sys.stderr)

    print(f"Chat model: {config.CHAT_MODEL_ID}  ·  Embedding: {config.EMBED_MODEL_ID}")
    print(f"Ready — {pipeline.size} chunks indexed. Mode: {mode}. Type :help for commands.\n")

    history: List[Tuple[str, str]] = []
    while True:
        try:
            raw = input(f"Q[{mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() in QUIT_WORDS:
            break
        if raw.startswith(":"):
            mode = _handle_command(raw[1:].strip().lower(), mode, history)
            continue
        chit_chat = smalltalk.reply(raw)
        if chit_chat is not None:
            # Not a grounded turn: reply and don't add it to the conversation memory.
            print(f"\n{chit_chat}\n")
            continue
        if stream:
            print()  # newline before the streamed answer
            on_token = lambda tok: print(tok, end="", flush=True)  # noqa: E731
            result = pipeline.answer(raw, history=history, mode=mode, on_token=on_token)
            print("\n")  # close the streamed answer
        else:
            result = pipeline.answer(raw, history=history, mode=mode)
            print(f"\n{result.answer}")
        _print_meta(result, debug)
        history.append((raw, result.answer))

    print(random.choice(GOODBYE_MESSAGES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline CAN-bus / firmware Q&A assistant.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print retrieved chunks and similarity scores with each answer",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(config.ANSWER_MODES),
        default=config.DEFAULT_MODE,
        help="answer style: 'short' (1-2 sentences) or 'explain' (fuller). Default: short.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="print each answer all at once instead of streaming it token-by-token",
    )
    args = parser.parse_args()
    run(debug=args.debug, mode=args.mode, stream=config.STREAM_DEFAULT and not args.no_stream)


if __name__ == "__main__":
    main()
