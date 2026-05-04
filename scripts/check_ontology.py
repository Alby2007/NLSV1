import json
from collections import Counter

with open("data/ontology.json") as f:
    ont = json.load(f)

paths = Counter(tuple(p["type_path"]) for p in ont)
print("=== Type path distribution (top 30) ===")
for path, count in sorted(paths.most_common(30), key=lambda x: x[0]):
    print(f"  {' > '.join(path):50s} {count}")

known_subtypes = {
    "ConcreteEntity", "AbstractEntity",
    "CausalRelation", "TemporalRelation", "SpatialRelation", "LogicalRelation",
    "NumericProperty", "QualitativeProperty",
    "CognitiveProcess", "PhysicalProcess",
    "Quantifier", "Connective", "Comparator",
    "Numeric",
    "Epistemic", "Deontic", "Operator",
    # top-level types are also valid leaf paths
    "Entity", "Relation", "Property", "Process", "Logical", "Numeric", "Modal",
}

print("\n=== Unknown/invented subtypes ===")
unknown = set()
for path in paths:
    for node in path:
        if node not in known_subtypes:
            unknown.add(node)
if unknown:
    for u in sorted(unknown):
        print(f"  UNKNOWN: {u}")
else:
    print("  None — all subtypes valid")

print(f"\n=== Reclassified primitives (40 total) ===")
reclassified = [p for p in ont if p.get("reclassified")]
for p in reclassified[:15]:
    print(f"  {p['symbol']:35s} orig={p['type']:12s} path={p['type_path']}")
if len(reclassified) > 15:
    print(f"  ... and {len(reclassified) - 15} more")
