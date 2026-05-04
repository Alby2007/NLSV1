import json

with open("data/grammar_signatures.json") as f:
    sigs = json.load(f)

critical = [
    "LOGICAL_IMPLIES", "CAUSAL_ENABLES", "TEMPORAL_BEFORE",
    "THEREFORE", "FORALL", "EXISTS", "LOGICAL_AND",
    "LOGICAL_NOT", "LOGICAL_OR", "BECAUSE", "ASSUMES",
    "KNOWN_TRUE", "ARITH_ADD", "ARITH_EQUALS",
]
lookup = {s["symbol"]: s for s in sigs}
print(f"{'SYMBOL':25s} {'ARITY':7s} {'DOMAIN':45s} RANGE")
print("-" * 90)
for sym in critical:
    s = lookup.get(sym)
    if s:
        print(f"{sym:25s} {s['arity']:<7} {str(s['domain']):45s} {s['range']}")
    else:
        print(f"{sym:25s} NOT FOUND")
