import json
from pathlib import Path

ont = json.loads(Path('data/ontology.json').read_text(encoding='utf-8'))
prims = {p['symbol']: p for p in ont}

check = [
    'TEMPORAL_REFERENCE', 'ARITH_LESS_THAN', 'ARITH_GREATER_THAN',
    'ARITH_EQUALS', 'ARITH_EQUAL', 'ARITH_ASSIGN', 'ARITH_MINUS',
    'ARITH_SUBTRACT', 'NUMERIC_UNKNOWN', 'ARITH_RESULT', 'ARITH_ADD',
    'ARITH_MULTIPLY', 'ARITH_DIVIDE', 'TEMPORAL_CHANGE', 'TEMPORAL_ORDER',
]

print("=== Exact matches ===")
for name in check:
    if name in prims:
        p = prims[name]
        status = "EXISTS "
        print(f"  {status}  {name:35s} type={p.get('type','?')}  arity={p.get('arity','?')}")
    else:
        print(f"  MISSING  {name:35s}")

print("\n=== All ARITH_* primitives ===")
for name, p in sorted(prims.items()):
    if name.startswith('ARITH_'):
        print(f"  {name:40s} type={p.get('type','?')}  arity={p.get('arity','?')}")

print("\n=== All TEMPORAL_* primitives ===")
for name, p in sorted(prims.items()):
    if name.startswith('TEMPORAL_'):
        print(f"  {name:40s} type={p.get('type','?')}  arity={p.get('arity','?')}")

print("\n=== All NUMERIC_* primitives ===")
for name, p in sorted(prims.items()):
    if name.startswith('NUMERIC_'):
        print(f"  {name:40s} type={p.get('type','?')}  arity={p.get('arity','?')}")
