"""
Diagnose: (1) NL leakage false positives, (2) bare-HEAD patterns still slipping through auto_repair.
"""
import json, re
from pathlib import Path

examples = [json.loads(l) for l in Path("data/corpus/train.jsonl").read_text(encoding="utf-8").strip().splitlines()]

# --- NL leakage: show ALL flagged examples with the offending words highlighted ---
print("=== NL LEAKAGE CASES ===")
nl_re_prose = re.compile(r'\b[a-z]{4,}\b')
nl_re_quoted = re.compile(r'"[^"]{4,}"')
leak_count = 0
for e in examples:
    chain = e.get("neuralese_chain", "")
    quoted = nl_re_quoted.findall(chain)
    words = [w for w in nl_re_prose.findall(chain) if w not in ('true', 'false', 'null')]
    if quoted or len(words) > 3:
        leak_count += 1
        if leak_count <= 10:
            print(f"  Chain: {chain[:120]}")
            print(f"  Prose words: {words[:10]}")
            print()

print(f"Total leaks: {leak_count}\n")

# --- Bare HEAD: find chains that still have unparenthesized applications ---
# Pattern: SYMBOL followed by space and then number or another SYMBOL NOT preceded by (
print("=== BARE HEAD PATTERNS IN WRITTEN CORPUS ===")
bare_head_re = re.compile(r'(?<!\()([A-Z][A-Z_]{2,})\s+(\d[\d.]*|[A-Z][A-Z_]+)\s+(\d[\d.]*|[A-Z][A-Z_]+)(?!\s*\))')
bare_count = 0
for e in examples:
    chain = e.get("neuralese_chain", "")
    hits = bare_head_re.findall(chain)
    if hits:
        bare_count += 1
        if bare_count <= 8:
            print(f"  {chain[:120]}")
            print(f"  Hits: {hits[:3]}")
            print()

print(f"Total with bare HEADs in corpus: {bare_count}/{len(examples)}")
