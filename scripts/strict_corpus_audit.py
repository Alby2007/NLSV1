"""
Audit the training corpus under STRICT type checking (domain/range rules).

The corpus was built under loose validation (arity + symbol existence only).
This measures how much of it would fail the strict check — which decides
whether strict parse validity is reportable at eval, or just a diagnostic.

Writes outputs/strict_audit.json and prints a summary.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_grammar.parser import validate, validate_strict, get_signatures
from config import TRAIN_CORPUS_PATH, OUTPUTS_DIR

OUT = OUTPUTS_DIR / "strict_audit.json"


def main():
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_CORPUS_PATH
    sigs = get_signatures()

    n = loose_ok = strict_ok = 0
    err_counts: Counter = Counter()
    head_err_counts: Counter = Counter()
    examples = []

    for line in open(corpus, encoding="utf-8"):
        item = json.loads(line)
        chain = item["neuralese_chain"]
        n += 1

        ok_loose, _ = validate(chain, sigs)
        ok_strict, err = validate_strict(chain, sigs)
        loose_ok += ok_loose
        strict_ok += ok_strict

        if ok_loose and not ok_strict and err:
            kind = err.split(":")[0]
            err_counts[kind] += 1
            if kind == "TYPE_ERROR":
                # "TYPE_ERROR: HEAD argN expects D, got T" → count by HEAD
                head = err.split()[1] if len(err.split()) > 1 else "?"
                head_err_counts[head] += 1
            if len(examples) < 10:
                examples.append({"question": item["question"][:80], "err": err, "chain": chain[:150]})

    report = {
        "corpus": str(corpus),
        "n": n,
        "loose_valid": loose_ok,
        "strict_valid": strict_ok,
        "strict_valid_rate": round(strict_ok / max(n, 1), 4),
        "newly_failing_by_error": dict(err_counts.most_common()),
        "top_type_error_heads": dict(head_err_counts.most_common(15)),
        "examples": examples,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"corpus: {corpus}  n={n}")
    print(f"loose valid:  {loose_ok} ({loose_ok/n:.1%})")
    print(f"strict valid: {strict_ok} ({strict_ok/n:.1%})")
    if err_counts:
        print("newly failing:", dict(err_counts.most_common()))
        print("top heads:", dict(head_err_counts.most_common(10)))
    print(f"written → {OUT}")


if __name__ == "__main__":
    main()
