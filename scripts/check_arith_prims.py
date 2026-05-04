import json

prims = {p["symbol"]: p for p in json.load(open("data/ontology.json"))}
sigs = {s["symbol"]: s for s in json.load(open("data/grammar_signatures.json"))}

check = [
    "MATHEMATICAL_CONCEPT_SUBTRACTION", "MATHEMATICAL_CONCEPT_SQUARE_ROOT",
    "NUMERIC_QUANTITY", "NUMERIC_FRACTION_REFERENCE",
    "MATHEMATICAL_CONCEPTS", "INVESTIGATE_STUDY_MATHEMATICAL_CONCEPTS",
    "MATHEMATICAL_CONCEPT_SQUARE_ROOT",
]
for sym in check:
    p = prims.get(sym)
    s = sigs.get(sym)
    arity = s["arity"] if s else "N/A"
    print(f"{sym}: ontology={p is not None} sigs={s is not None} arity={arity}")

# Also show all Process/Numeric primitives that Qwen might use for arithmetic
print("\nNumeric primitives in ontology:")
nums = [p for p in prims.values() if p.get("type") == "Numeric"]
for p in sorted(nums, key=lambda x: x["symbol"])[:20]:
    s = sigs.get(p["symbol"])
    print(f"  {p['symbol']:40s} arity={s['arity'] if s else 'NO_SIG'}")
