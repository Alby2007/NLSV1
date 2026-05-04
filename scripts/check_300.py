"""
300-line checkpoint check — measures post-fix quality on examples written after index 183.
Checks: unknown symbol rate, mean/max compression, lambda over-nesting.
"""
import json
import re
from pathlib import Path

CORPUS = Path("data/corpus/train.jsonl")
PRIMS  = Path("data/ontology.json")
POST_FIX_INDEX = 183  # examples written before the #var fix

with open(PRIMS, encoding="utf-8") as f:
    valid_symbols = {p["symbol"] for p in json.load(f) if p.get("symbol")}

examples = []
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

total = len(examples)
recent = examples[POST_FIX_INDEX:]

if len(recent) < 10:
    print(f"Only {total} total, {len(recent)} post-fix — not enough yet. Come back later.")
    raise SystemExit(0)

ratios = [e["compression_ratio"] for e in recent]
lambda_count = sum(1 for e in recent if e["neuralese_chain"].count("λ") > 2)

unknown_count = 0
unknown_freq: dict[str, int] = {}
for e in recent:
    chain = e["neuralese_chain"]
    found = set(re.findall(r"\b[A-Z][A-Z_]+[A-Z]\b", chain))
    unknown = found - valid_symbols
    if unknown:
        unknown_count += 1
    for sym in unknown:
        unknown_freq[sym] = unknown_freq.get(sym, 0) + 1

mean_r = sum(ratios) / len(ratios)
max_r  = max(ratios)
unknown_pct = 100 * unknown_count / len(recent)
lambda_pct  = 100 * lambda_count  / len(recent)

print(f"Post-fix examples (index {POST_FIX_INDEX}+): {len(recent)}")
print(f"")
print(f"Compression  mean={mean_r:.3f}  max={max_r:.3f}    target: mean≤0.65, no 1.5+")
print(f"Unknown syms {unknown_count}/{len(recent)} ({unknown_pct:.0f}%)          target: <15%")
print(f"Lambda nest  {lambda_count}/{len(recent)} ({lambda_pct:.0f}%)          target: <10%")

# Verdict
ok_compression = mean_r <= 0.65 and max_r <= 1.5
ok_unknown     = unknown_pct < 15
ok_lambda      = lambda_pct  < 10

print()
print(f"Compression  {'PASS' if ok_compression else 'FAIL'}")
print(f"Unknown syms {'PASS' if ok_unknown     else 'FAIL'}")
print(f"Lambda nest  {'PASS' if ok_lambda      else 'FAIL'}")

if ok_compression and ok_unknown and ok_lambda:
    print(f"\n✓ ALL CLEAR — safe to run to completion")
else:
    print(f"\n✗ ISSUES REMAIN — review top offenders below")

if unknown_freq:
    print(f"\nTop unknown symbols (gap candidates):")
    for sym, cnt in sorted(unknown_freq.items(), key=lambda x: -x[1])[:20]:
        marker = " ◄ CANDIDATE" if cnt >= 5 else ""
        print(f"  {sym:35s} {cnt:4d}x{marker}")
