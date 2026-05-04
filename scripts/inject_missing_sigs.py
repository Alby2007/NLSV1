"""
Inject hand-crafted signatures for Modal and Numeric seeded primitives
that were excluded from generate_signatures.py (only Relation/Process/Logical
were included in SIGNABLE_TYPES).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRAMMAR_SIGNATURES_PATH

MISSING_SIGS = [
    # Modal — unary wrappers around a claim
    {"symbol": "KNOWN_TRUE",    "arity": 1, "domain": ["Logical|Relation|Entity"], "range": "Modal"},
    {"symbol": "BELIEVED_TRUE", "arity": 1, "domain": ["Logical|Relation|Entity"], "range": "Modal"},
    {"symbol": "UNCERTAIN",     "arity": 1, "domain": ["Logical|Relation|Entity"], "range": "Modal"},
    {"symbol": "KNOWN_FALSE",   "arity": 1, "domain": ["Logical|Relation|Entity"], "range": "Modal"},

    # Numeric arithmetic — binary
    {"symbol": "ARITH_ADD",      "arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Numeric"},
    {"symbol": "ARITH_SUBTRACT", "arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Numeric"},
    {"symbol": "ARITH_MULTIPLY", "arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Numeric"},
    {"symbol": "ARITH_DIVIDE",   "arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Numeric"},
    {"symbol": "ARITH_EQUALS",   "arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Logical"},
    {"symbol": "ARITH_MODULO",   "arity": 2, "domain": ["Numeric|Entity", "Numeric|Entity"], "range": "Numeric"},
]

with open(GRAMMAR_SIGNATURES_PATH, encoding="utf-8") as f:
    sigs = json.load(f)

existing = {s["symbol"] for s in sigs}
injected = 0
for sig in MISSING_SIGS:
    if sig["symbol"] not in existing:
        sigs.append(sig)
        injected += 1
        print(f"Injected: {sig['symbol']} arity={sig['arity']}")
    else:
        print(f"Already present: {sig['symbol']}")

with open(GRAMMAR_SIGNATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(sigs, f, indent=2)

print(f"\nTotal signatures: {len(sigs)} (+{injected} injected)")
