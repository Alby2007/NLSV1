"""
Patch ontology.json and grammar_signatures.json with missing primitives.
Run once: python scripts/patch_ontology.py
"""
import json
from pathlib import Path

ONT_PATH = Path('data/ontology.json')
SIG_PATH = Path('data/grammar_signatures.json')

ont = json.loads(ONT_PATH.read_text(encoding='utf-8'))
sigs = json.loads(SIG_PATH.read_text(encoding='utf-8'))

existing_symbols = {p['symbol'] for p in ont}
existing_sig_symbols = {s['symbol'] for s in sigs}

# ── New primitives to add ────────────────────────────────────────────────
NEW_PRIMITIVES = [
    {
        'symbol': 'ARITH_LESS_THAN',
        'type': 'Logical',
        'subtype': 'NumericRelation',
        'gloss': 'numeric less-than comparison between two values',
        'centroid': None,
        'confidence': 1.0,
    },
    {
        'symbol': 'ARITH_GREATER_THAN',
        'type': 'Logical',
        'subtype': 'NumericRelation',
        'gloss': 'numeric greater-than comparison between two values',
        'centroid': None,
        'confidence': 1.0,
    },
    {
        'symbol': 'ARITH_ASSIGN',
        'type': 'Process',
        'subtype': 'CognitiveProcess',
        'gloss': 'assign a computed numeric value to a named quantity or variable',
        'centroid': None,
        'confidence': 1.0,
    },
]

# ── New grammar signatures ───────────────────────────────────────────────
NEW_SIGNATURES = [
    {
        'symbol': 'ARITH_LESS_THAN',
        'arity': 2,
        'arg_types': ['Numeric', 'Numeric'],
        'return_type': 'Logical',
        'description': '(ARITH_LESS_THAN a b) — true when a < b',
    },
    {
        'symbol': 'ARITH_GREATER_THAN',
        'arity': 2,
        'arg_types': ['Numeric', 'Numeric'],
        'return_type': 'Logical',
        'description': '(ARITH_GREATER_THAN a b) — true when a > b',
    },
    {
        'symbol': 'ARITH_ASSIGN',
        'arity': 2,
        'arg_types': ['Entity', 'Numeric'],
        'return_type': 'Process',
        'description': '(ARITH_ASSIGN #label value) — bind a value to a named quantity',
    },
]

# ── Apply patches ────────────────────────────────────────────────────────
added_ont = []
for p in NEW_PRIMITIVES:
    if p['symbol'] not in existing_symbols:
        ont.append(p)
        added_ont.append(p['symbol'])
    else:
        print(f"  SKIP (exists): {p['symbol']}")

added_sig = []
for s in NEW_SIGNATURES:
    if s['symbol'] not in existing_sig_symbols:
        sigs.append(s)
        added_sig.append(s['symbol'])
    else:
        print(f"  SKIP sig (exists): {s['symbol']}")

ONT_PATH.write_text(json.dumps(ont, indent=2, ensure_ascii=False), encoding='utf-8')
SIG_PATH.write_text(json.dumps(sigs, indent=2, ensure_ascii=False), encoding='utf-8')

print(f"\nOntology: added {len(added_ont)} primitives: {added_ont}")
print(f"Signatures: added {len(added_sig)} signatures: {added_sig}")
print(f"Ontology total: {len(ont)}")
print(f"Signatures total: {len(sigs)}")
