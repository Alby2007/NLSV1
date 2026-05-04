"""
Inject hand-seeded primitives into data/primitives.json before Phase 2.

- Skips any symbol already present (won't overwrite mined primitives)
- Removes disabled symbols that are too vague for stable grammar signatures
- Assigns centroid_embedding: null (fine-tune embedding layer learns from usage)
- Assigns confidence: 1.0 (hand-verified)
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PRIMITIVES_PATH
from scripts.seed_primitives import SEED_PRIMITIVES, DISABLED_SYMBOLS


def main():
    with open(PRIMITIVES_PATH, encoding="utf-8") as f:
        primitives = json.load(f)

    print(f"Loaded {len(primitives)} primitives.")

    # Remove disabled symbols
    before = len(primitives)
    primitives = [p for p in primitives if p["symbol"] not in DISABLED_SYMBOLS]
    removed = before - len(primitives)
    if removed:
        print(f"Removed {removed} disabled primitives: {DISABLED_SYMBOLS}")

    existing_symbols = {p["symbol"] for p in primitives}
    next_id = max(int(p["id"][1:]) for p in primitives) + 1

    injected = 0
    skipped = 0
    for seed in SEED_PRIMITIVES:
        if seed["symbol"] in existing_symbols:
            skipped += 1
            continue
        primitives.append({
            "id": f"P{next_id:04d}",
            "symbol": seed["symbol"],
            "type": seed["type"],
            "gloss": seed["gloss"],
            "centroid_embedding": None,
            "confidence": 1.0,
            "example_tokens": [],
            "source": "hand_seeded",
        })
        existing_symbols.add(seed["symbol"])
        next_id += 1
        injected += 1

    with open(PRIMITIVES_PATH, "w", encoding="utf-8") as f:
        json.dump(primitives, f, indent=2)

    print(f"Injected {injected} seed primitives ({skipped} already present).")
    print(f"Total primitives: {len(primitives)}")

    types = Counter(p["type"] for p in primitives)
    print("\nType distribution:")
    for t, c in types.most_common():
        print(f"  {t:30s} {c}")


if __name__ == "__main__":
    main()
