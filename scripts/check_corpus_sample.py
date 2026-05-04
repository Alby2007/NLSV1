"""
Quick quality check on the first N examples in train.jsonl.
Run once 100+ lines are written:
    python scripts/check_corpus_sample.py
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CORPUS_PATH = Path("data/corpus/train.jsonl")
SAMPLE_N = 500
PRINT_EXAMPLES = 3

def has_nl_leak(chain: str) -> bool:
    """Detect quoted strings or lowercase prose leaking into Neuralese.
    #var references (e.g. #result, #total) are valid grammar and must not be flagged.
    """
    if re.search(r'"[^"]{4,}"', chain):
        return True
    # Strip valid #var tokens before checking for prose
    scrubbed = re.sub(r'#[a-z][a-z0-9_]*', '', chain)
    words = re.findall(r'\b[a-z]{4,}\b', scrubbed)
    nl_words = [w for w in words if w not in ('true', 'false', 'null')]
    return len(nl_words) > 2  # more than 2 unexplained lowercase words = likely prose

def main():
    if not CORPUS_PATH.exists():
        print("No corpus file yet.")
        sys.exit(1)

    examples = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                examples.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
            if len(examples) >= SAMPLE_N:
                break

    if not examples:
        print("File exists but no valid JSON lines yet.")
        sys.exit(1)

    print(f"Sampled {len(examples)} examples from corpus.\n")

    ratios = [e["compression_ratio"] for e in examples]
    print(f"Compression ratio:")
    print(f"  Mean : {sum(ratios)/len(ratios):.3f}")
    print(f"  Min  : {min(ratios):.3f}")
    print(f"  Max  : {max(ratios):.3f}")
    print(f"  >1.5 (would discard): {sum(1 for r in ratios if r > 1.5)}")
    print(f"  <=0.65 (target)     : {sum(1 for r in ratios if r <= 0.65)}/{len(ratios)}")

    chains = [e.get("neuralese_chain", "") for e in examples]
    nl_leaks = [c for c in chains if has_nl_leak(c)]
    print(f"\nNL leakage (quoted strings or prose in chain): {len(nl_leaks)}/{len(chains)}")
    if nl_leaks:
        print("  Examples of leakage:")
        for c in nl_leaks[:2]:
            print(f"    {c[:150]}")

    with open("data/ontology.json", encoding="utf-8") as f:
        valid_symbols = {p["symbol"] for p in json.load(f) if p.get("symbol")}

    unknown_symbol_count = 0
    unknown_examples = []
    for e in examples:
        chain = e.get("neuralese_chain", "")
        found_symbols = set(re.findall(r'\b[A-Z][A-Z_]+[A-Z]\b', chain))
        unknown = found_symbols - valid_symbols
        if unknown:
            unknown_symbol_count += 1
            unknown_examples.append((unknown, e["question"][:60]))

    print(f"\nUnknown symbols (hallucinated, not in ontology): {unknown_symbol_count}/{len(examples)}")
    if unknown_examples[:3]:
        print("  Sample unknown symbols:")
        for syms, q in unknown_examples[:3]:
            print(f"    {syms} | Q: {q}")

    unknown_freq = Counter()
    for e in examples:
        chain = e.get("neuralese_chain", "")
        found = set(re.findall(r'\b[A-Z][A-Z_]+[A-Z]\b', chain))
        unknown_freq.update(found - valid_symbols)

    print(f"\nTop unknown symbols (ontology gap candidates — appear 5+ times):")
    gap_candidates = [(sym, cnt) for sym, cnt in unknown_freq.most_common(30) if cnt >= 2]
    for sym, count in gap_candidates[:20]:
        marker = " ◄ CANDIDATE" if count >= 5 else ""
        print(f"  {sym:35s} {count:4d}x{marker}")

    print(f"\n--- {PRINT_EXAMPLES} random examples ---")
    for e in random.sample(examples, min(PRINT_EXAMPLES, len(examples))):
        print(f"Q: {e['question'][:80]}")
        print(f"Neuralese: {e['neuralese_chain'][:250]}")
        print(f"Ratio: {e['compression_ratio']:.3f}")
        print()

if __name__ == "__main__":
    main()
