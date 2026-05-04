"""
Scan pipeline.log for ARITY_MISMATCH and UNKNOWN_SYMBOL patterns,
tally the top offenders to guide signature fixes.
"""
import re
from collections import Counter
from pathlib import Path

log = Path("outputs/pipeline.log").read_text(encoding="utf-8", errors="ignore")

arity_re = re.compile(r"ARITY_MISMATCH: (\w+) expects (\d+) args, got (\d+)")
unknown_re = re.compile(r"UNKNOWN_SYMBOL: (\w+)")

arity_mismatches = Counter()
unknown_symbols = Counter()

for m in arity_re.finditer(log):
    sym, expected, got = m.group(1), m.group(2), m.group(3)
    arity_mismatches[f"{sym} (expects {expected}, got {got})"] += 1

for m in unknown_re.finditer(log):
    unknown_symbols[m.group(1)] += 1

print("Top ARITY_MISMATCH offenders:")
for item, count in arity_mismatches.most_common(15):
    print(f"  {count:4d}x  {item}")

print("\nTop UNKNOWN_SYMBOL offenders (invented by Qwen):")
for sym, count in unknown_symbols.most_common(20):
    print(f"  {count:4d}x  {sym}")
