import json, pathlib

NEW_SYMBOLS = [
    ("DIVIDE", "Numeric", "Arithmetic", "Divides one quantity by another", 2),
    ("PLUS", "Numeric", "Arithmetic", "Adds two quantities together", 2),
    ("EQUALS", "Numeric", "Arithmetic", "Asserts two quantities are equal", 2),
    ("MINUS", "Numeric", "Arithmetic", "Subtracts one quantity from another", 2),
    ("MULTIPLY", "Numeric", "Arithmetic", "Multiplies two quantities together", 2),
    ("FEET", "Numeric", "Measurement", "Distance measured in feet", 1),
    ("HOURS", "Numeric", "Measurement", "Duration measured in hours", 1),
    ("REVENUE", "Numeric", "Financial", "Total income from sales or services", 1),
    ("YEAR", "Numeric", "Temporal", "Duration or index of one calendar year", 1),
    ("TOTAL_DAYS", "Numeric", "Temporal", "Total number of days in a period", 1),
    ("MINUTES", "Numeric", "Measurement", "Duration measured in minutes", 1),
    ("MILES", "Numeric", "Measurement", "Distance measured in miles", 1),
    ("POUNDS", "Numeric", "Measurement", "Weight measured in pounds", 1),
    ("DOLLARS", "Numeric", "Financial", "Monetary amount in dollars", 1),
    ("ITEMS", "Numeric", "Quantity", "Count of discrete items", 1),
    ("STUDENTS", "Numeric", "Quantity", "Count of students", 1),
    ("WORKERS", "Numeric", "Quantity", "Count of workers or employees", 1),
    ("DAYS", "Numeric", "Temporal", "Duration measured in days", 1),
    ("WEEKS", "Numeric", "Temporal", "Duration measured in weeks", 1),
    ("MONTHS", "Numeric", "Temporal", "Duration measured in months", 1),
]

base = pathlib.Path(__file__).parent.parent / "data"

for fname in ("ontology.json", "primitives.json"):
    data = json.loads((base / fname).read_text())
    existing = {p["symbol"] for p in data}
    added = 0
    for sym, typ, sub, gloss, _ in NEW_SYMBOLS:
        if sym not in existing:
            data.append({"symbol": sym, "type": typ, "subtype": sub, "gloss": gloss, "centroid": None, "confidence": 1.0})
            added += 1
    (base / fname).write_text(json.dumps(data, indent=2))
    print(f"{fname}: +{added}, total {len(data)}")

sigs = json.loads((base / "grammar_signatures.json").read_text())
existing_s = {s["symbol"] for s in sigs}
added_s = 0
for sym, typ, sub, gloss, arity in NEW_SYMBOLS:
    if sym not in existing_s:
        sigs.append({"symbol": sym, "arity": arity, "domain": [typ] * arity, "range": typ})
        added_s += 1
(base / "grammar_signatures.json").write_text(json.dumps(sigs, indent=2))
print(f"grammar_signatures.json: +{added_s}, total {len(sigs)}")
