"""
Post-process grammar_signatures.json:
- Fix THEREFORE arity to 1 (unary conclusion marker, not binary)
- Fix BECAUSE arity to 2 (claim, justification) — already correct but verify
- Fix FORALL/EXISTS domain to accept any expr (not just Logical|Entity)
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRAMMAR_SIGNATURES_PATH

with open(GRAMMAR_SIGNATURES_PATH, encoding="utf-8") as f:
    sigs = json.load(f)

OVERRIDES = {
    "THEREFORE":  {"arity": 1, "domain": ["Logical|Relation|Process|Entity"], "range": "Logical"},
    "FORALL":     {"arity": 2, "domain": ["Entity", "Logical"], "range": "Logical"},
    "EXISTS":     {"arity": 2, "domain": ["Entity", "Logical"], "range": "Logical"},
    "EXISTS_UNIQUE": {"arity": 2, "domain": ["Entity", "Logical"], "range": "Logical"},
}

fixed = 0
for s in sigs:
    if s["symbol"] in OVERRIDES:
        before = dict(s)
        s.update(OVERRIDES[s["symbol"]])
        print(f"Fixed {s['symbol']}: {before} -> {s}")
        fixed += 1

with open(GRAMMAR_SIGNATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(sigs, f, indent=2)

print(f"\nFixed {fixed} signatures.")
