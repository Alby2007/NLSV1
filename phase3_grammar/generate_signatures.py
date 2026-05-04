"""
Phase 3: Generate arity + type signatures for all Relation and Process primitives.

Calls Ollama with the full primitive list to propose signatures,
then saves data/grammar_signatures.json.
Also asks Ollama to identify grammar gaps.
"""

import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ONTOLOGY_PATH, GRAMMAR_SIGNATURES_PATH
from utils.ollama_client import get_client
from utils.logging import get_logger
from utils.phases import require_phase, mark_complete

logger = get_logger(__name__)

BATCH_SIZE = 20
SIGNABLE_TYPES = {"Relation", "Process", "Logical", "Numeric"}


def generate_signatures_batch(client, primitives: list[dict]) -> list[dict]:
    items = [{"symbol": p["symbol"], "type": p["type"], "gloss": p["gloss"]} for p in primitives]
    prompt = (
        "For each of the following Neuralese primitives, propose an arity and type signature.\n"
        "Rules:\n"
        "  - arity: number of arguments (int, 0–4)\n"
        "  - domain: list of argument type constraints, one entry per argument. "
        "    Each entry is a '|'-separated union of type names from: "
        "[Entity, Relation, Property, Process, Logical, Numeric, Modal, ConcreteEntity, "
        "AbstractEntity, CausalRelation, TemporalRelation, SpatialRelation, LogicalRelation, "
        "NumericProperty, QualitativeProperty, CognitiveProcess, PhysicalProcess, "
        "Quantifier, Connective, Comparator, Cardinal, Ordinal, Operator, Epistemic, Deontic]\n"
        "  - range: the output type of the primitive\n\n"
        f"Primitives:\n{json.dumps(items, indent=2)}\n\n"
        "Return JSON array only:\n"
        '[{"symbol": "...", "arity": N, "domain": ["Type1|Type2", ...], "range": "Type"}]'
    )

    parsed = client.chat_json(prompt, context_label="signatures_batch")
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = list(parsed.values())[0] if parsed else []
    return parsed if isinstance(parsed, list) else []


def identify_grammar_gaps(client, primitives: list[dict]) -> str:
    summary = [{"symbol": p["symbol"], "type": p["type"], "gloss": p["gloss"]} for p in primitives[:200]]
    prompt = (
        "You are helping design a formal reasoning language called Neuralese v1.\n"
        "The grammar supports: S-expressions, typed lambda abstraction (λ x:T . expr), "
        "confidence assertions {expr | 0.0-1.0}, and sequences [expr expr].\n\n"
        f"Here is a sample of {len(summary)} primitives:\n{json.dumps(summary, indent=2)}\n\n"
        "Identify:\n"
        "1. Any semantic patterns in these primitives that the grammar CANNOT express\n"
        "2. Suggested grammar extensions (each must not introduce ambiguity)\n\n"
        "Respond in plain text."
    )
    result = client.chat_text(prompt, context_label="grammar_gaps")
    return result or "Gap analysis returned no response."


def main():
    require_phase(2)
    logger.info("Phase 3 — generate_signatures starting")

    with open(ONTOLOGY_PATH, encoding="utf-8") as f:
        primitives = json.load(f)

    signable = [p for p in primitives if p.get("type") in SIGNABLE_TYPES]
    logger.info(f"Generating signatures for {len(signable)} signable primitives ({', '.join(SIGNABLE_TYPES)})...")

    # Incremental: load existing signatures and skip already-signed symbols
    existing_sigs = []
    if GRAMMAR_SIGNATURES_PATH.exists():
        with open(GRAMMAR_SIGNATURES_PATH, encoding="utf-8") as f:
            existing_sigs = json.load(f)
    already_signed = {s["symbol"] for s in existing_sigs}
    todo = [p for p in signable if p["symbol"] not in already_signed]
    logger.info(f"  {len(already_signed)} already signed, {len(todo)} remaining.")

    client = get_client()
    all_signatures = list(existing_sigs)

    for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Signature batches"):
        batch = todo[i:i + BATCH_SIZE]
        results = generate_signatures_batch(client, batch)
        all_signatures.extend(results)

    known_symbols = {s["symbol"] for s in all_signatures}
    missing = [p["symbol"] for p in signable if p["symbol"] not in known_symbols]
    if missing:
        print(f"  {len(missing)} primitives missing signatures — filling with defaults")
        for sym in missing:
            all_signatures.append({"symbol": sym, "arity": 2, "domain": ["Entity|Process", "Entity|Process"], "range": "Relation"})

    with open(GRAMMAR_SIGNATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_signatures, f, indent=2)
    logger.info(f"Saved {len(all_signatures)} signatures to {GRAMMAR_SIGNATURES_PATH}")

    logger.info("Running grammar gap analysis...")
    gaps = identify_grammar_gaps(client, primitives)
    gap_path = GRAMMAR_SIGNATURES_PATH.parent / "grammar_gaps.txt"
    gap_path.write_text(gaps, encoding="utf-8")
    logger.info(f"Gap analysis saved to {gap_path}")
    logger.info("\n--- GRAMMAR GAPS ---\n" + gaps[:1000])
    mark_complete(3, f"{len(all_signatures)} signatures generated")


if __name__ == "__main__":
    main()
