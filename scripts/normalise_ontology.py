"""
Post-process ontology.json:
1. Replace invented subtype names with valid ones (fallback to first valid subtype for the top type)
2. Fix hand-seeded primitives that got misclassified (ARITH_* must stay Numeric, LOGICAL_* Logical, etc.)
3. Drop noise primitives that have no semantic value
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ONTOLOGY_PATH, TYPE_HIERARCHY

VALID_SUBTYPES = {subtype for subtypes in TYPE_HIERARCHY.values() for subtype in subtypes}
VALID_TOP_TYPES = set(TYPE_HIERARCHY.keys())

# Primitives to drop — noise that slipped past confidence filter
DROP_SYMBOLS = {
    "SEMANTIC_VAGUE", "UNCLEAR_REQUEST", "QUASAR_OR_QUARTIC",
    "CONTEXTUAL_INCONSISTENCY", "CONTEXTUAL_WITH_PHRASE", "ABSTRACT_SYMBOL",
    "COMMON_CONCEPT_EXAMPLE", "STANDARD_CONCEPT",
}

# Force correct type_path for hand-seeded primitives that got misclassified
FORCE_TYPE_PATH = {
    # Arithmetic ops must be Numeric
    "ARITH_ADD":      ["Numeric", "NumericProperty"],
    "ARITH_SUBTRACT": ["Numeric", "NumericProperty"],
    "ARITH_MULTIPLY": ["Numeric", "NumericProperty"],
    "ARITH_DIVIDE":   ["Numeric", "NumericProperty"],
    "ARITH_EQUALS":   ["Numeric", "NumericProperty"],
    "ARITH_MODULO":   ["Numeric", "NumericProperty"],
    # Logical connectives must be Logical
    "LOGICAL_AND":    ["Logical", "Connective"],
    "LOGICAL_OR":     ["Logical", "Connective"],
    "LOGICAL_NOT":    ["Logical", "Connective"],
    "LOGICAL_IMPLIES":["Logical", "Connective"],
    "LOGICAL_IFF":    ["Logical", "Connective"],
    "LOGICAL_XOR":    ["Logical", "Connective"],
    "FORALL":         ["Logical", "Quantifier"],
    "EXISTS":         ["Logical", "Quantifier"],
    "EXISTS_UNIQUE":  ["Logical", "Quantifier"],
    "THEREFORE":      ["Logical", "Connective"],
    "BECAUSE":        ["Logical", "Connective"],
    "CONTRADICTS":    ["Logical", "Comparator"],
    "ASSUMES":        ["Logical", "Connective"],
    "GIVEN":          ["Logical", "Connective"],
    # Modal
    "KNOWN_TRUE":     ["Modal", "Epistemic"],
    "BELIEVED_TRUE":  ["Modal", "Epistemic"],
    "UNCERTAIN":      ["Modal", "Epistemic"],
    "KNOWN_FALSE":    ["Modal", "Epistemic"],
    # Relations
    "CAUSAL_ENABLES":     ["Relation", "CausalRelation"],
    "CAUSAL_PREVENTS":    ["Relation", "CausalRelation"],
    "CAUSAL_REQUIRES":    ["Relation", "CausalRelation"],
    "CAUSAL_CONTRIBUTES": ["Relation", "CausalRelation"],
    "TEMPORAL_BEFORE":    ["Relation", "TemporalRelation"],
    "TEMPORAL_AFTER":     ["Relation", "TemporalRelation"],
    "TEMPORAL_DURING":    ["Relation", "TemporalRelation"],
    "TEMPORAL_UNTIL":     ["Relation", "TemporalRelation"],
    "IS_INSTANCE_OF":     ["Relation", "LogicalRelation"],
    "IS_PART_OF":         ["Relation", "LogicalRelation"],
    "IS_DEFINED_AS":      ["Relation", "LogicalRelation"],
    "IS_EQUIVALENT_TO":   ["Relation", "LogicalRelation"],
    "IS_GREATER_THAN":    ["Relation", "LogicalRelation"],
    "IS_LESS_THAN":       ["Relation", "LogicalRelation"],
    "IS_OPPOSITE_OF":     ["Relation", "LogicalRelation"],
}


def fix_type_path(entry: dict) -> dict | None:
    symbol = entry["symbol"]

    if symbol in DROP_SYMBOLS:
        return None

    if symbol in FORCE_TYPE_PATH:
        entry["type_path"] = FORCE_TYPE_PATH[symbol]
        entry["type"] = FORCE_TYPE_PATH[symbol][0]
        entry["reclassified"] = False
        return entry

    path = entry.get("type_path", [])
    if not path:
        entry["type_path"] = [entry["type"]]
        return entry

    top = path[0]
    if top not in VALID_TOP_TYPES:
        # Top type itself is invalid — fall back to entry's original type
        top = entry["type"]
        path = [top]

    if len(path) == 1:
        return entry  # just top type, fine

    subtype = path[1]
    if subtype not in VALID_SUBTYPES:
        # Invented subtype — replace with first valid subtype for this top type
        valid = TYPE_HIERARCHY.get(top, [])
        path = [top, valid[0]] if valid else [top]
        entry["type_path"] = path

    return entry


def main():
    with open(ONTOLOGY_PATH, encoding="utf-8") as f:
        ontology = json.load(f)

    print(f"Loaded {len(ontology)} entries.")

    cleaned = []
    dropped = 0
    fixed_paths = 0

    for entry in ontology:
        original_path = entry.get("type_path", [])
        result = fix_type_path(entry)
        if result is None:
            dropped += 1
            continue
        if result["type_path"] != original_path:
            fixed_paths += 1
        cleaned.append(result)

    with open(ONTOLOGY_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)

    print(f"Dropped {dropped} noise primitives.")
    print(f"Fixed {fixed_paths} invalid type paths.")
    print(f"Final ontology: {len(cleaned)} primitives.")

    # Verify no unknown subtypes remain
    from collections import Counter
    paths = Counter(tuple(p["type_path"]) for p in cleaned)
    unknown = []
    for path in paths:
        for node in path:
            if node not in VALID_SUBTYPES and node not in VALID_TOP_TYPES:
                unknown.append((node, path))
    if unknown:
        print(f"\nWARNING — still have unknown nodes:")
        for node, path in unknown:
            print(f"  {node} in {path}")
    else:
        print("\nAll type paths valid.")

    print("\nFinal type distribution:")
    type_counts = Counter(p["type_path"][0] for p in cleaned)
    for t, c in type_counts.most_common():
        print(f"  {t:30s} {c}")


if __name__ == "__main__":
    main()
