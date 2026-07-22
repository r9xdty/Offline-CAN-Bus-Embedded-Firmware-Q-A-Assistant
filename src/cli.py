"""CLI — Phase 1 interface (spec §10.1).

    python -m src.cli            # ask questions in a loop
    python -m src.cli --debug    # also print retrieved chunks + similarity scores

Type `quit` or submit an empty line to exit.
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .pipeline import Answer, Pipeline


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
    print(f"Ready — {pipeline.size} chunks indexed. Ask a CAN-bus / firmware question.\n")

    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit", "bye", "q", "hadi sg", "thank you for your services", "thank you for your soul"}:
            break
        result = pipeline.answer(question)
        _print_answer(result, debug)

    goodbye_messages = ["Bye", "hadi sende sg", "Adamsin Quershma", "Gorusuruz knk", "babays"]
    print(goodbye_messages(randomize()))


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
