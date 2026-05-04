"""
Inject NUMERIC_RESULT and RUNNING_TOTAL into both ontology and signatures.
These give Qwen valid symbols for arithmetic result references,
eliminating the RESULT/TOTAL hallucination pattern.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ONTOLOGY_PATH, GRAMMAR_SIGNATURES_PATH

NEW_PRIMITIVES = [
    {
        "symbol": "NUMERIC_RESULT",
        "type": "Numeric",
        "gloss": "the numeric output of the immediately preceding computation",
        "centroid": None,
        "confidence": 1.0,
        "subtype": "Numeric",
    },
    {
        "symbol": "RUNNING_TOTAL",
        "type": "Numeric",
        "gloss": "accumulated sum across multiple arithmetic steps",
        "centroid": None,
        "confidence": 1.0,
        "subtype": "Numeric",
    },
]

NEW_SIGNATURES = [
    {"symbol": "NUMERIC_RESULT",  "arity": 0, "domain": [], "range": "Numeric"},
    {"symbol": "RUNNING_TOTAL",   "arity": 0, "domain": [], "range": "Numeric"},
]

# --- ontology ---
with open(ONTOLOGY_PATH, encoding="utf-8") as f:
    prims = json.load(f)
existing_syms = {p["symbol"] for p in prims}
added_prims = [p for p in NEW_PRIMITIVES if p["symbol"] not in existing_syms]
prims.extend(added_prims)
with open(ONTOLOGY_PATH, "w", encoding="utf-8") as f:
    json.dump(prims, f, indent=2)

# --- signatures ---
with open(GRAMMAR_SIGNATURES_PATH, encoding="utf-8") as f:
    sigs = json.load(f)
existing_sig_syms = {s["symbol"] for s in sigs}
added_sigs = [s for s in NEW_SIGNATURES if s["symbol"] not in existing_sig_syms]
sigs.extend(added_sigs)
with open(GRAMMAR_SIGNATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(sigs, f, indent=2)

print(f"Ontology: added {len(added_prims)} primitives → {len(prims)} total")
print(f"Signatures: added {len(added_sigs)} signatures → {len(sigs)} total")
for p in added_prims:
    print(f"  + {p['symbol']} ({p['type']}): {p['gloss']}")
