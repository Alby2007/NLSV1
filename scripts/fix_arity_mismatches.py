"""
Fix arity mismatches identified from pipeline.log analysis.
- TEMPORAL_CHANGE: used as 1, 2, or 3 args → make variadic (arity=0 = variadic in our convention)
  But since grammar enforces exact arity, set to 2 and let Qwen adapt.
  Actually the real fix: TEMPORAL_CHANGE describes a state change, should be arity=2 (from, to).
  Qwen uses it as (TEMPORAL_CHANGE state) [1 arg] or (TEMPORAL_CHANGE from to by) [3 args].
  Most natural is arity=2 (from_state, to_state). Keep as 2 — retries will teach Qwen.

- DURATION_CALCULATION: arity=1 but used as 4 → fix to arity=2 (duration, unit)
- QUANTITY_CALCULATION: arity=1 but used as 5 → fix to arity=2 (operand, operand)
- NEW_ITEM_OR_EVENT: arity=2 but used as 5 → fix to arity=2 (keep, Qwen overloads)
- TOTAL_NUMBER_CALCULATION: arity=2 but used as 4/6 → fix to arity=2 (keep)
- GET_RESULT: arity=1 but used as 3 → fix to arity=2 (expression, result)
- ARITH_ADD/SUBTRACT/MULTIPLY/DIVIDE: arity=2, Qwen sometimes uses as 1 or 3.
  These are correct at arity=2. Leave them — Qwen errors on these should self-correct via retries.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRAMMAR_SIGNATURES_PATH

FIXES = {
    "DURATION_CALCULATION": {"arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Numeric"},
    "QUANTITY_CALCULATION":  {"arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Numeric"},
    "GET_RESULT":            {"arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Numeric"},
}

with open(GRAMMAR_SIGNATURES_PATH, encoding="utf-8") as f:
    sigs = json.load(f)

fixed = 0
for s in sigs:
    if s["symbol"] in FIXES:
        before = dict(s)
        s.update(FIXES[s["symbol"]])
        print(f"Fixed {s['symbol']}: arity {before['arity']} → {s['arity']}")
        fixed += 1

with open(GRAMMAR_SIGNATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(sigs, f, indent=2)

print(f"\n{fixed} signatures fixed.")
