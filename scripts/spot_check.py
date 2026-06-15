"""
Manual spot-check of 100 randomly sampled corpus examples.

Presents each example interactively in the terminal.
You categorize failures with a single keypress, or mark as OK.

Failure categories:
  1  WRONG_ARITH      — arithmetic in the chain is incorrect
  2  HALLUC_PRIM      — uses a made-up primitive not in the ontology
  3  GRAMMAR_INVALID  — structurally malformed (shouldn't parse but slipped through)
  4  CORRECT_VERBOSE  — correct but longer than it needed to be (not a hard fail)
  5  SEMANTICALLY_OFF — chain doesn't reflect the question's reasoning
  o  OK               — looks good
  s  SKIP             — unsure, skip this example
  q  QUIT             — save and quit early

Results written to outputs/spot_check_results.json.

Usage:
    python scripts/spot_check.py [--n 100] [--seed 42] [--corpus data/corpus/train.jsonl]
    python scripts/spot_check.py --resume   (continue from where you left off)
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_CORPUS_PATH, OUTPUTS_DIR
from phase3_grammar.parser import validate, get_signatures

RESULTS_PATH = OUTPUTS_DIR / "spot_check_results.json"

CATEGORIES = {
    "1": "WRONG_ARITH",
    "2": "HALLUC_PRIM",
    "3": "GRAMMAR_INVALID",
    "4": "CORRECT_VERBOSE",
    "5": "SEMANTICALLY_OFF",
    "o": "OK",
    "s": "SKIP",
}

CATEGORY_HELP = (
    "  [1] WRONG_ARITH      arithmetic error in chain\n"
    "  [2] HALLUC_PRIM      made-up primitive (not in ontology)\n"
    "  [3] GRAMMAR_INVALID  structurally malformed\n"
    "  [4] CORRECT_VERBOSE  correct but needlessly long\n"
    "  [5] SEMANTICALLY_OFF chain doesn't match the question's logic\n"
    "  [o] OK               looks good\n"
    "  [s] SKIP             unsure\n"
    "  [q] QUIT             save and exit\n"
)


def load_ontology_symbols() -> set[str]:
    ontology_path = Path("data/ontology.json")
    if not ontology_path.exists():
        return set()
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    return {p["symbol"] for p in ontology}


def highlight_hallucinations(chain: str, legit: set[str]) -> str:
    """Mark hallucinated UPPER_CASE tokens with [!] for easy spotting."""
    def replace(m):
        tok = m.group(0)
        if tok in legit or re.match(r'^\d', tok):
            return tok
        return f"[!{tok}!]"
    return re.sub(r'[A-Z][A-Z_]+[A-Z]', replace, chain)


def display_example(idx: int, total: int, ex: dict, legit_prims: set[str], sigs) -> None:
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"Example {idx}/{total}  |  compression={ex.get('compression_ratio', '?')}  "
          f"steps={ex.get('step_count', '?')}  source={ex.get('source', '?')}")
    print(sep)
    print(f"Q: {ex['question']}")
    print(f"\nGold answer: {ex['answer']}")
    print(f"\nOriginal CoT:\n  {ex.get('original_cot', '')[:300]}")
    print(f"\nNeuralese chain:")
    chain = ex.get("neuralese_chain", "")
    highlighted = highlight_hallucinations(chain, legit_prims)
    # Wrap long chains
    if len(highlighted) > 200:
        print(f"  {highlighted[:200]}")
        print(f"  ... [{len(highlighted)-200} more chars]")
    else:
        print(f"  {highlighted}")

    # Auto-flags
    ok, parse_err = validate(chain, sigs)
    flags = []
    if not ok:
        flags.append(f"GRAMMAR: {parse_err}")
    halluc = [t for t in re.findall(r'[A-Z][A-Z_]+[A-Z]', chain) if t not in legit_prims]
    if halluc:
        flags.append(f"HALLUC: {list(set(halluc))[:5]}")
    if ex.get("compression_ratio", 0) > 1.0:
        flags.append(f"EXPANSION (ratio={ex['compression_ratio']:.2f})")
    if flags:
        print(f"\n  AUTO-FLAGS: {' | '.join(flags)}")

    print(f"\n{CATEGORY_HELP}")


def save_results(results: list[dict], counts: Counter) -> None:
    output = {
        "total_reviewed": len(results),
        "category_counts": dict(counts),
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of examples to review.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--corpus", type=str, default=str(TRAIN_CORPUS_PATH))
    parser.add_argument("--resume", action="store_true", help="Resume from saved results.")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    raw = [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    rng = random.Random(args.seed)
    sample = rng.sample(raw, min(args.n, len(raw)))

    sigs = get_signatures()
    legit_prims = load_ontology_symbols()
    print(f"Loaded {len(legit_prims)} ontology primitives for hallucination detection.")

    # Resume support
    existing_results: list[dict] = []
    reviewed_questions: set[str] = set()
    if args.resume and RESULTS_PATH.exists():
        saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        existing_results = saved.get("results", [])
        reviewed_questions = {r["question"] for r in existing_results}
        print(f"Resuming: {len(existing_results)} already reviewed.")

    results = list(existing_results)
    counts: Counter = Counter(r["category"] for r in results)

    to_review = [ex for ex in sample if ex["question"] not in reviewed_questions]
    total_to_do = len(to_review)

    print(f"\nReviewing {total_to_do} examples. Press key + Enter to categorize.")
    print("Tip: [!TOKEN!] = hallucinated primitive not in ontology.\n")

    for i, ex in enumerate(to_review, start=len(existing_results) + 1):
        display_example(i, len(existing_results) + total_to_do, ex, legit_prims, sigs)

        while True:
            try:
                raw_input = input("Category: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted — saving progress.")
                save_results(results, counts)
                sys.exit(0)

            if raw_input == "q":
                print("Quitting — saving progress.")
                save_results(results, counts)
                _print_summary(counts, results)
                sys.exit(0)

            if raw_input not in CATEGORIES:
                print(f"Unknown key '{raw_input}'. Use: {list(CATEGORIES.keys())}")
                continue

            category = CATEGORIES[raw_input]
            note = ""
            if category not in ("OK", "SKIP"):
                try:
                    note = input("Optional note (Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    pass

            record = {
                "question": ex["question"],
                "answer": ex["answer"],
                "compression_ratio": ex.get("compression_ratio"),
                "source": ex.get("source", "unknown"),
                "category": category,
                "note": note,
                "chain_preview": ex.get("neuralese_chain", "")[:150],
            }
            results.append(record)
            counts[category] += 1
            print(f"  → {category}" + (f": {note}" if note else ""))
            break

        # Autosave every 10
        if i % 10 == 0:
            save_results(results, counts)

    save_results(results, counts)
    _print_summary(counts, results)


def _print_summary(counts: Counter, results: list[dict]) -> None:
    total = len(results)
    print("\n" + "=" * 50)
    print(f"SPOT-CHECK SUMMARY ({total} reviewed)")
    print("=" * 50)
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        print(f"  {cat:20s}: {count:4d} ({pct:.1f}%)")

    # Compute OK rate excluding SKIPs
    non_skip = total - counts.get("SKIP", 0)
    ok = counts.get("OK", 0) + counts.get("CORRECT_VERBOSE", 0)
    if non_skip > 0:
        print(f"\n  Quality rate (OK+verbose / non-skip): {ok}/{non_skip} = {100*ok/non_skip:.1f}%")

    # Surface most common failure notes
    notes = [r["note"] for r in results if r.get("note") and r["category"] not in ("OK", "SKIP")]
    if notes:
        print(f"\n  Flagged notes:")
        for n in notes[:10]:
            print(f"    - {n}")

    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
