"""
Phase 4: Corpus statistics and coverage reporting.
"""

import json
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_CORPUS_PATH, CORPUS_STATS_PATH, ONTOLOGY_PATH


def main():
    with open(TRAIN_CORPUS_PATH, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    with open(ONTOLOGY_PATH, encoding="utf-8") as f:
        primitives = json.load(f)
    all_symbols = {p["symbol"] for p in primitives}

    ratios = [e["compression_ratio"] for e in examples]
    step_counts = [e.get("step_count", 1) for e in examples]

    used_symbols: Counter = Counter()
    for e in examples:
        for sym in re.findall(r"[A-Z][A-Z0-9_]+", e["neuralese_chain"]):
            if sym in all_symbols:
                used_symbols[sym] += 1

    coverage = len(used_symbols) / max(len(all_symbols), 1)

    print("=" * 50)
    print("CORPUS STATISTICS")
    print("=" * 50)
    print(f"Total examples:          {len(examples)}")
    print(f"Compression ratio:")
    print(f"  Mean:                  {sum(ratios)/len(ratios):.3f}")
    print(f"  Min:                   {min(ratios):.3f}")
    print(f"  Max:                   {max(ratios):.3f}")
    print(f"  ≤ 0.65 (target):       {sum(1 for r in ratios if r <= 0.65)/len(ratios):.1%}")
    print(f"Steps per example:")
    print(f"  Mean:                  {sum(step_counts)/len(step_counts):.1f}")
    print(f"  Min:                   {min(step_counts)}")
    print(f"  Max:                   {max(step_counts)}")
    print(f"Primitive coverage:      {coverage:.1%}  ({len(used_symbols)}/{len(all_symbols)} symbols used)")
    print(f"\nTop 20 most-used primitives:")
    for sym, cnt in used_symbols.most_common(20):
        print(f"  {sym}: {cnt}")

    unused = all_symbols - set(used_symbols.keys())
    print(f"\nUnused primitives: {len(unused)}")
    if unused:
        print("  Sample:", list(unused)[:10])

    stats = {
        "total_examples": len(examples),
        "mean_compression_ratio": round(sum(ratios) / len(ratios), 4),
        "min_compression_ratio": round(min(ratios), 4),
        "max_compression_ratio": round(max(ratios), 4),
        "fraction_at_target": round(sum(1 for r in ratios if r <= 0.65) / len(ratios), 4),
        "primitive_coverage": round(coverage, 4),
        "used_symbol_count": len(used_symbols),
        "total_symbol_count": len(all_symbols),
        "top_symbols": dict(used_symbols.most_common(50)),
    }

    if CORPUS_STATS_PATH.exists():
        with open(CORPUS_STATS_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(stats)
        stats = existing

    with open(CORPUS_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {CORPUS_STATS_PATH}")


if __name__ == "__main__":
    main()
