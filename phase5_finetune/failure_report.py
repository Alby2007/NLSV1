"""
Phase 5: Failure analysis — cluster errors, trace root causes, feed back to earlier phases.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_RESULTS_PATH, OUTPUTS_DIR
from phase3_grammar.parser import validate, get_signatures

REPORT_PATH = OUTPUTS_DIR / "failure_report.json"


def categorise_failure(item: dict, sigs) -> str:
    """Return the failure category for one eval result."""
    if not item["parse_valid"]:
        reasoning = item.get("reasoning", "")
        ok, err = validate(reasoning, sigs)
        if err:
            return err.split(":")[0]
        return "PARSE_ERROR"
    if not item["correct"]:
        return "WRONG_ANSWER"
    return "CORRECT"


def trace_step_error(reasoning: str, sigs) -> str | None:
    """Find the first invalid step in a sequence chain."""
    steps = re.findall(r"\(.*?\)|\{.*?\}|\[.*?\]", reasoning)
    for step in steps:
        ok, err = validate(step, sigs)
        if not ok:
            return f"Step '{step[:50]}' → {err}"
    return None


def main():
    if not EVAL_RESULTS_PATH.exists():
        print(f"No eval results found at {EVAL_RESULTS_PATH}. Run eval.py first.")
        sys.exit(1)

    with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
        eval_results = json.load(f)

    sigs = get_signatures()
    report = {}

    for dataset_name, result in eval_results.items():
        if "sample_results" not in result:
            continue

        samples = result["sample_results"]
        categories: Counter = Counter()
        grammar_gaps: list[str] = []
        wrong_answer_traces: list[dict] = []

        for item in samples:
            cat = categorise_failure(item, sigs)
            categories[cat] += 1

            if cat != "CORRECT":
                trace = trace_step_error(item.get("reasoning", ""), sigs)
                if trace:
                    grammar_gaps.append(trace)
                if cat == "WRONG_ANSWER":
                    wrong_answer_traces.append({
                        "question": item["question"][:100],
                        "gold": item["gold"],
                        "predicted": item["predicted_answer"],
                        "reasoning_preview": item.get("reasoning", "")[:150],
                    })

        report[dataset_name] = {
            "category_counts": dict(categories),
            "accuracy": result.get("accuracy"),
            "parse_validity": result.get("parse_validity_rate"),
            "grammar_gap_traces": grammar_gaps[:10],
            "wrong_answer_samples": wrong_answer_traces[:5],
            "iteration_signals": [],
        }

        signals = report[dataset_name]["iteration_signals"]
        if categories.get("PARSE_ERROR", 0) > len(samples) * 0.1:
            signals.append("High parse error rate → review Phase 3 grammar or retrain with tighter prompts")
        if categories.get("UNKNOWN_SYMBOL", 0) > 0:
            signals.append("UNKNOWN_SYMBOL errors → Phase 1 primitive coverage gap, re-run with more corpus")
        if categories.get("ARITY_MISMATCH", 0) > 0:
            signals.append("ARITY_MISMATCH errors → Phase 3 signature table incorrect, regenerate signatures")
        if categories.get("WRONG_ANSWER", 0) > len(samples) * 0.15:
            signals.append("High wrong-answer rate → Phase 4 translation quality may be poor; check compression ratio")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Failure report saved to {REPORT_PATH}")

    print("\n=== FAILURE SUMMARY ===")
    for ds, r in report.items():
        print(f"\n{ds}:")
        for cat, count in r["category_counts"].items():
            print(f"  {cat}: {count}")
        for sig in r.get("iteration_signals", []):
            print(f"  → {sig}")


if __name__ == "__main__":
    main()
