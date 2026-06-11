import json
from pathlib import Path
from collections import Counter
import re

# ── 1. Failure reason breakdown from pipeline log ──────────────────────────
log = Path('outputs/pipeline.log').read_text(encoding='utf-8', errors='replace')
lines = log.splitlines()

start_idx = next(i for i, l in enumerate(lines) if 'Phase 4' in l and 'translate' in l.lower())
trans_lines = lines[start_idx:]

fail_pat = re.compile(r'attempt (\d)/3 failed: (\w+)', re.IGNORECASE)
failure_by_attempt = {1: Counter(), 2: Counter(), 3: Counter()}

for l in trans_lines:
    m = fail_pat.search(l)
    if m:
        attempt = int(m.group(1))
        ftype = m.group(2)
        failure_by_attempt[attempt][ftype] += 1

print("=== Failure reasons by attempt ===")
for attempt in [1, 2, 3]:
    total = sum(failure_by_attempt[attempt].values())
    label = "final (discard)" if attempt == 3 else f"attempt {attempt}"
    print(f"\n  [{label}] — {total} total")
    for k, v in failure_by_attempt[attempt].most_common():
        print(f"    {k:30s}: {v:5d}  ({100*v/max(total,1):.1f}%)")

total_discard = sum(failure_by_attempt[3].values())
print(f"\n  Total examples exhausting all 3 retries: {total_discard}")
print(f"  Total GSM8K input: 7473")
print(f"  Valid corpus: 6417")
print(f"  Discarded (retry-exhausted): {7473 - 6417} = {7473-6417}")

# Check for unknown symbol details
unknown_syms = Counter()
for l in trans_lines:
    if 'UNKNOWN_SYMBOL' in l:
        m2 = re.search(r'UNKNOWN_SYMBOL: (\S+)', l)
        if m2:
            unknown_syms[m2.group(1)] += 1
print(f"\n=== Top 20 unknown symbols causing failures ===")
for sym, cnt in unknown_syms.most_common(20):
    print(f"  {sym:35s}: {cnt}")

# ── 2. Compression by difficulty ──────────────────────────────────────────
corpus = [json.loads(l) for l in Path('data/corpus/train.jsonl').read_text(encoding='utf-8').splitlines()]

buckets = {'easy (<60w)': [], 'medium (60-120w)': [], 'hard (>120w)': []}
for ex in corpus:
    cot_len = len(ex['original_cot'].split())
    if cot_len < 60:
        buckets['easy (<60w)'].append(ex['compression_ratio'])
    elif cot_len < 120:
        buckets['medium (60-120w)'].append(ex['compression_ratio'])
    else:
        buckets['hard (>120w)'].append(ex['compression_ratio'])

print("\n=== Compression ratio by difficulty bucket ===")
for bucket, ratios in buckets.items():
    if ratios:
        mean = sum(ratios) / len(ratios)
        std = (sum((r - mean)**2 for r in ratios) / len(ratios)) ** 0.5
        print(f"  {bucket:20s}  n={len(ratios):4d}  mean={mean:.3f}  std={std:.3f}  min={min(ratios):.3f}  max={max(ratios):.3f}")

# ── 3. Top 50 worst-compressing valid examples ────────────────────────────
top50 = sorted(corpus, key=lambda x: x['compression_ratio'], reverse=True)[:50]

print("\n=== Top 50 highest compression ratio examples ===")
print(f"{'#':3s}  {'Ratio':5s}  {'CoT_w':5s}  {'NL_w':5s}  Question")
for i, ex in enumerate(top50):
    cot_w = len(ex['original_cot'].split())
    nl_w = len(ex['neuralese_chain'].split())
    print(f"  {i+1:2d}   {ex['compression_ratio']:.3f}   {cot_w:4d}    {nl_w:4d}   {ex['question'][:70]}")

print("\n=== Sample neuralese chains from top 5 worst ===")
for ex in top50[:5]:
    print(f"\n  Q: {ex['question'][:80]}")
    print(f"  Ratio: {ex['compression_ratio']:.3f}")
    print(f"  CoT:  {ex['original_cot'][:120]}")
    print(f"  NL:   {ex['neuralese_chain'][:200]}")
