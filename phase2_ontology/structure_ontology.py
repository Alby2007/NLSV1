"""
Phase 2: Organise primitives into a typed hierarchy via Ollama.

Reads data/primitives.json, calls Ollama per type bucket,
assigns each primitive a full type_path, saves data/ontology.json.
"""

import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TYPE_HIERARCHY, PRIMITIVES_PATH, ONTOLOGY_PATH
from utils.ollama_client import get_client
from utils.logging import get_logger
from utils.phases import require_phase, mark_complete

logger = get_logger(__name__)


def structure_bucket(client, top_type: str, subtypes: list[str], primitives: list[dict]) -> list[dict]:
    """Ask Ollama to assign subtypes and flag reclassifications for one type bucket."""
    items = [{"id": p["id"], "symbol": p["symbol"], "gloss": p["gloss"]} for p in primitives]
    prompt = (
        f"Here are {len(items)} primitives of type '{top_type}', each with a symbol and gloss.\n"
        f"Assign each to the most specific subtype from this list: {subtypes}.\n"
        "Also flag any primitive that clearly belongs in a DIFFERENT top-level type than currently assigned "
        f"(top-level types: {list(TYPE_HIERARCHY.keys())}).\n\n"
        f"Primitives:\n{json.dumps(items, indent=2)}\n\n"
        "Return a JSON array only, no preamble:\n"
        '[{"id": "...", "subtype": "...", "reclassify_to": null}]'
    )

    parsed = client.chat_json(prompt, context_label=f"ontology_{top_type}")
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = list(parsed.values())[0] if parsed else []
    return parsed if isinstance(parsed, list) else []


def main():
    require_phase(1)
    logger.info("Phase 2 — structure_ontology starting")

    with open(PRIMITIVES_PATH, encoding="utf-8") as f:
        primitives = json.load(f)

    logger.info(f"Loaded {len(primitives)} primitives.")
    client = get_client()

    by_type: dict[str, list[dict]] = {}
    for p in primitives:
        by_type.setdefault(p["type"], []).append(p)

    id_to_primitive = {p["id"]: p for p in primitives}
    assignments: dict[str, dict] = {}

    for top_type, subtypes in tqdm(TYPE_HIERARCHY.items(), desc="Structuring types"):
        bucket = by_type.get(top_type, [])
        if not bucket:
            continue
        print(f"\n  {top_type}: {len(bucket)} primitives → subtypes {subtypes}")

        results = structure_bucket(client, top_type, subtypes, bucket)

        for r in results:
            pid = r.get("id")
            if pid not in id_to_primitive:
                continue
            assignments[pid] = {
                "subtype": r.get("subtype", subtypes[0]),
                "reclassify_to": r.get("reclassify_to"),
            }

    ontology = []
    for p in primitives:
        pid = p["id"]
        asgn = assignments.get(pid, {})
        top_type = p["type"]
        subtype = asgn.get("subtype", TYPE_HIERARCHY.get(top_type, ["Unknown"])[0])
        reclassify = asgn.get("reclassify_to")
        # Normalise: Ollama sometimes returns ["Entity"] instead of "Entity"
        if isinstance(reclassify, list):
            reclassify = reclassify[0] if reclassify else None

        if reclassify and reclassify in TYPE_HIERARCHY:
            effective_top = reclassify
            type_path = [reclassify]
        else:
            effective_top = top_type
            type_path = [top_type, subtype] if subtype else [top_type]

        entry = {**p, "type_path": type_path, "reclassified": bool(reclassify)}
        entry.pop("centroid_embedding", None)
        ontology.append(entry)

    with open(ONTOLOGY_PATH, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2)

    logger.info(f"Saved {len(ontology)} primitives to {ONTOLOGY_PATH}")

    reclassified = sum(1 for e in ontology if e["reclassified"])
    logger.info(f"Reclassified: {reclassified} primitives")
    type_counts = {}
    for e in ontology:
        type_counts[e["type_path"][0]] = type_counts.get(e["type_path"][0], 0) + 1
    for t, c in sorted(type_counts.items()):
        logger.info(f"  {t}: {c}")
    mark_complete(2, f"{len(ontology)} primitives structured")


if __name__ == "__main__":
    main()
