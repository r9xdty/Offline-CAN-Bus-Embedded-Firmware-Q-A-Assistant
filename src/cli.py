"""CLI — Phase 1 interface (spec §10.1), with conversation memory and answer modes.

    python -m src.cli                    # ask questions in a loop (remembers the conversation)
    python -m src.cli --debug            # also print retrieved chunks + similarity scores
    python -m src.cli --mode explain     # start in "explain" mode (default: short)

In-session commands:
    :short / :explain    switch answer mode
    :general on / off    toggle the general-knowledge fallback tier (bare :general flips it)
    :reset (:clear)      forget the conversation so far
    :examples (:ex)      show sample questions
    :help                show commands

Type `quit` (or `q`, `bye`, an empty line, Ctrl+C) to exit.

Output is colorized with plain ANSI codes when stdout is an interactive terminal (disabled
automatically when piped/captured, or when the NO_COLOR env var is set — see no-color.org).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Tuple

from . import config, foundry_client, smalltalk
from .pipeline import Answer, Pipeline

# --------------------------------------------------------------------------- #
# Minimal ANSI color helper — no dependency, auto-disabled for non-tty/NO_COLOR.
# --------------------------------------------------------------------------- #
GREEN = "32"
YELLOW = "33"
AMBER = "33"  # no distinct "orange" in the base 8-color palette; yellow reads as amber
RED = "31"
DIM = "2"
BOLD = "1"


def _use_color() -> bool:
    """Decide at print time (not import time) so tests that swap sys.stdout see plain text."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    return getattr(sys.stdout, "isatty", lambda: False)()


def _c(text: str, code: str) -> str:
    """Wrap `text` in ANSI SGR `code` when color is enabled; otherwise return it unchanged."""
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


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
    "  :general on/off     toggle the general-knowledge fallback (bare :general flips it)\n"
    "  :reset (:clear)     forget the conversation so far\n"
    "  :examples (:ex)     show sample questions\n"
    "  :help               show this help\n"
    "  quit / q / bye      exit"
)

_GENERAL_NOTICE = "(i) General knowledge — not grounded in your documents."


def _kind_color(kind: str) -> str:
    return {"grounded": GREEN, "general": AMBER, "refusal": DIM}.get(kind, "")


def _score_color(score: float) -> str:
    if score >= 0.5:
        return GREEN
    if score >= 0.25:
        return YELLOW
    return RED


def _print_meta(result: Answer, debug: bool) -> None:
    """Print the sources + stats line (and, with --debug, the retrieved chunks)."""
    if result.sources:
        print(f"Sources: {_c(str(result.sources), GREEN)}")
    else:
        print(_c("Sources: []", DIM))
    if result.top_score is not None:
        top = _c(f"{result.top_score:.2f}", _score_color(result.top_score))
    else:
        top = "n/a"
    kind = _c(result.kind, _kind_color(result.kind))
    print(f"({result.elapsed_s:.1f}s · {result.mode} · {kind} · top match {top})")
    if debug:
        print("\n--- retrieved chunks ---")
        if not result.chunks:
            print("  (none)")
        for ch in result.chunks:
            preview = ch.content.replace("\n", " ")
            score = _c(f"score={ch.score:.4f}", DIM)
            print(f"  {score}  [{ch.source}#{ch.chunk_index}]  {preview[:140]}...")
    print()


def _handle_command(cmd: str, mode: str, history: List[Tuple[str, str]]) -> str:
    """Apply an in-session `:command`. Returns the (possibly changed) mode."""
    if cmd in config.ANSWER_MODES:
        print(f"[mode: {cmd}]")
        return cmd
    if cmd in {"reset", "clear", "new"}:
        history.clear()
        print("[conversation cleared]")
    elif cmd in {"examples", "ex"}:
        print("Example questions:")
        for i, q in enumerate(config.EXAMPLE_QUESTIONS, start=1):
            print(f"  {i}. {q}")
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

    general_on = config.GENERAL_KNOWLEDGE_ENABLED
    print(f"Chat model: {config.CHAT_MODEL_ID}  ·  Embedding: {config.EMBED_MODEL_ID}")
    print(
        f"Ready — {pipeline.size} chunks indexed. Mode: {mode}. "
        f"General knowledge: {'on' if general_on else 'off'}. Type :help for commands."
    )
    print("Tip: type :examples to see sample questions, :help for commands.\n")

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
            cmd = raw[1:].strip().lower()
            if cmd == "general" or cmd.startswith("general "):
                arg = cmd[len("general"):].strip()
                if arg in ("", "toggle"):
                    general_on = not general_on
                elif arg in ("on", "true", "1"):
                    general_on = True
                elif arg in ("off", "false", "0"):
                    general_on = False
                else:
                    print(f"[unknown ':general' argument '{arg}' — try 'on' or 'off']")
                    continue
                state = "on" if general_on else "off"
                print(_c(f"[general knowledge: {state}]", AMBER if general_on else DIM))
                continue
            mode = _handle_command(cmd, mode, history)
            continue
        chit_chat = smalltalk.reply(raw)
        if chit_chat is not None:
            # Not a grounded turn: reply and don't add it to the conversation memory.
            print(f"\n{chit_chat}\n")
            continue
        if stream:
            print()  # newline before the streamed answer
            on_token = lambda tok: print(tok, end="", flush=True)  # noqa: E731
            result = pipeline.answer(
                raw, history=history, mode=mode, on_token=on_token, general_enabled=general_on,
            )
            print("\n")  # close the streamed answer
        else:
            result = pipeline.answer(raw, history=history, mode=mode, general_enabled=general_on)
            print(f"\n{result.answer}")
        if result.kind == "general":
            print(_c(_GENERAL_NOTICE, AMBER))
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
