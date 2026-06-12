"""
Round-2 discard analysis — reads retranslate_discards.log directly.
No re-processing needed.
"""
import re
from collections import Counter
from pathlib import Path
import json

log = Path('outputs/retranslate_discards.log').read_text(encoding='utf-8', errors='replace')
lines = log.splitlines()

fail_pat   = re.compile(r'attempt (\d)/3 failed: (\w+)', re.IGNORECASE)
unknown_pat = re.compile(r'UNKNOWN_SYMBOL: (\S+)')
arity_pat   = re.compile(r'ARITY_MISMATCH: (\S+) expects')

failure_by_attempt = {1: Counter(), 2: Counter(), 3: Counter()}
unknown_syms  = Counter()
arity_syms    = Counter()

for l in lines:
    m = fail_pat.search(l)
    if m:
        attempt = int(m.group(1))
        ftype   = m.group(2)
        failure_by_attempt[attempt][ftype] += 1

        u = unknown_pat.search(l)
        if u:
            unknown_syms[u.group(1)] += 1

        a = arity_pat.search(l)
        if a:
            arity_syms[a.group(1)] += 1

print("=== Round-2 failure reasons by attempt ===")
for attempt in [1, 2, 3]:
    total = sum(failure_by_attempt[attempt].values())
    label = "final (discard)" if attempt == 3 else f"attempt {attempt}"
    print(f"\n  [{label}] — {total} total")
    for k, v in failure_by_attempt[attempt].most_common():
        print(f"    {k:30s}: {v:5d}  ({100*v/max(total,1):.1f}%)")

total_final = sum(failure_by_attempt[3].values())
print(f"\n  Total exhausted all retries: {total_final}")
print(f"  Recovered: 605 / 2220  ({100*605/2220:.1f}%)")
print(f"  Still failing: {2220 - 605}")

print("\n=== Top 20 unknown symbols (round 2) ===")
r1_top = 'TEMPORAL_REFERENCE'
for sym, cnt in unknown_syms.most_common(20):
    flag = " ← WAS #1 IN R1" if sym == r1_top else ""
    print(f"  {sym:40s}: {cnt:4d}{flag}")

print("\n=== Top 10 arity-mismatch symbols (round 2) ===")
for sym, cnt in arity_syms.most_common(10):
    print(f"  {sym:40s}: {cnt:4d}")

# Compare R1 vs R2 top unknown
print("\n=== R1 vs R2 unknown symbol comparison ===")
r1_counts = {
    'TEMPORAL_REFERENCE': 473, 'UNCERTAINTY_EXPRESSION': 56, 'NUMBER': 54,
    'X': 37, 'HAVE': 28, 'DURATION': 26, 'LET': 25, 'UNCERTAIN_EXPRESSION': 24,
    'HAS': 20, 'AGE': 17, 'ARITH_LESS_THAN': 16, 'ARITH_GREATER_THAN': 15,
    'I': 14, 'EAT': 14, 'GIVE': 13, 'ARITH_ASSIGN': 13,
    'VARIABLE': 13, 'ARITH_MINUS': 12, 'ARITH_RESULT': 12, 'ARITH_EQUAL': 11,
}
print(f"  {'Symbol':40s} {'R1':>6} {'R2':>6}  {'Change':>8}")
all_syms = sorted(set(list(r1_counts.keys()) + [s for s,_ in unknown_syms.most_common(20)]),
                  key=lambda s: -unknown_syms.get(s, 0))
for sym in all_syms[:20]:
    r1 = r1_counts.get(sym, 0)
    r2 = unknown_syms.get(sym, 0)
    change = f"{'↓' if r2 < r1 else '↑' if r2 > r1 else '='}{abs(r2-r1)}" if r1 else f"new:{r2}"
    print(f"  {sym:40s} {r1:6d} {r2:6d}  {change:>8}")
