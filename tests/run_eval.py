"""Evaluation runner (spec §12) — score the pipeline against tests/eval_set.jsonl.

    python -m tests.run_eval            # run all eval items through the real pipeline
    python -m tests.run_eval --json     # machine-readable per-item results

Each eval item is one of:
  - {"expected_behavior": "refuse"} -> passes when the answer is exactly the refusal string.
  - {"expected_behavior": "answer", "must_include_any": [...], "expect_source": "..."} ->
    passes when the answer is NOT a refusal and contains at least one required substring
    (case-insensitive). If `expect_source` is present it is reported but not required to pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

# Allow `python tests/run_eval.py` as well as `python -m tests.run_eval`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent / "eval_set.jsonl"


def load_eval(path: Path = EVAL_PATH) -> List[dict]:
    items: List[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def grade(item: dict, answer_text: str, sources: List[str]) -> tuple[bool, str]:
    """Return (passed, reason) for one graded item."""
    is_refusal = answer_text.strip() == config.REFUSAL_TEXT
    behavior = item.get("expected_behavior")

    if behavior == "refuse":
        if is_refusal:
            return True, "refused as expected"
        return False, "expected refusal but got an answer"

    # behavior == "answer"
    if is_refusal:
        return False, "expected an answer but got the refusal string"
    needles = [n.lower() for n in item.get("must_include_any", [])]
    if needles:
        hay = answer_text.lower()
        if not any(n in hay for n in needles):
            return False, f"answer missing all of {item.get('must_include_any')}"
    src_note = ""
    expect_source = item.get("expect_source")
    if expect_source:
        hit = expect_source in sources
        src_note = f" (expected source {expect_source}: {'cited' if hit else 'not cited'})"
    return True, "answer contains a required fact" + src_note


def run_eval(
    answer_fn: Optional[Callable[[str], "object"]] = None,
    items: Optional[List[dict]] = None,
) -> dict:
    """Run every eval item and return a results dict. `answer_fn(question) -> Answer`.

    Defaults to the real cached pipeline; inject a fake in tests.
    """
    if items is None:
        items = load_eval()
    if answer_fn is None:
        from src.pipeline import answer_query

        answer_fn = answer_query

    results = []
    passed = 0
    for item in items:
        question = item["question"]
        t0 = time.perf_counter()
        ans = answer_fn(question)
        dt = time.perf_counter() - t0
        answer_text = getattr(ans, "answer", str(ans))
        sources = getattr(ans, "sources", [])
        ok, reason = grade(item, answer_text, sources)
        passed += int(ok)
        results.append(
            {
                "question": question,
                "expected_behavior": item.get("expected_behavior"),
                "passed": ok,
                "reason": reason,
                "answer": answer_text,
                "sources": sources,
                "latency_s": round(dt, 3),
            }
        )
    return {"total": len(items), "passed": passed, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CAN-bus RAG eval set.")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    report = run_eval()

    if args.json:
        print(json.dumps(report, indent=2))
        return

    for r in report["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] ({r['latency_s']:.2f}s) {r['question']}")
        print(f"       -> {r['reason']}")
        if not r["passed"]:
            preview = r["answer"].replace("\n", " ")[:160]
            print(f"       answer: {preview}")
    print(f"\n{report['passed']}/{report['total']} passed")
    sys.exit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
