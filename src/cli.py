"""CLI — Phase 1 interface (spec §10.1).

    python -m src.cli            # ask questions in a loop
    python -m src.cli --debug    # also print retrieved chunks + similarity scores

Type `quit` (or `q`, `bye`, an empty line, Ctrl+C) to exit.
"""

from __future__ import annotations

import argparse
import random
import sys

from . import config, foundry_client
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


def _print_answer(result: Answer, debug: bool) -> None:
    print(f"\n{result.answer}")
    if result.sources:
        print(f"Sources: {result.sources}")
    else:
        print("Sources: []")
    if debug:
        print("\n--- retrieved chunks ---")
        if not result.chunks:
            print("  (none)")
        for ch in result.chunks:
            preview = ch.content.replace("\n", " ")
            print(f"  score={ch.score:.4f}  [{ch.source}#{ch.chunk_index}]  {preview[:140]}...")
    print()


def run(debug: bool = False) -> None:
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

    print(f"Ready — {pipeline.size} chunks indexed. Ask a CAN-bus / firmware question.\n")

    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in QUIT_WORDS:
            break
        result = pipeline.answer(question)
        _print_answer(result, debug)

    print(random.choice(GOODBYE_MESSAGES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline CAN-bus / firmware Q&A assistant.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print retrieved chunks and similarity scores with each answer",
    )
    args = parser.parse_args()
    run(debug=args.debug)


if __name__ == "__main__":
    main()
