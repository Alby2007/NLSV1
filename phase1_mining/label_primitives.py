"""
Phase 1 — Step 3: Label each centroid via Ollama (Qwen2.5).

For each centroid:
  1. Find top-K nearest tokens by cosine similarity
  2. Build context windows from metadata
  3. Call Ollama to get symbol/type/gloss/confidence
  4. Filter low-confidence, deduplicate near-identical centroids
  5. Save data/primitives.json
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NEAREST_TOKENS_K, CONTEXT_WINDOW,
    LABEL_MIN_CONFIDENCE, DEDUP_COSINE_THRESHOLD,
    DATA_DIR, PRIMITIVES_PATH,
    CENTROIDS_PATH, ASSIGNMENTS_PATH, ACTIVATION_META_PATH,
    PHASE1_FLAG,
)
from utils.ollama_client import get_client
from utils.logging import get_logger
from utils.phases import mark_complete

logger = get_logger(__name__)
META_PATH = ACTIVATION_META_PATH

VALID_TYPES = {"Entity", "Relation", "Property", "Process", "Logical", "Numeric", "Modal"}



def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (n, d), B: (m, d) → (n, m) cosine similarities."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T


def load_meta(path: Path) -> list[dict]:
    meta = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta


def build_context(meta_entry: dict, all_meta: list[dict]) -> str:
    sent_idx = meta_entry["sent_idx"]
    pos = meta_entry["pos"]
    window = [
        m["token_str"]
        for m in all_meta
        if m["sent_idx"] == sent_idx and abs(m["pos"] - pos) <= CONTEXT_WINDOW
    ]
    return " ".join(window)


def label_centroid(client, contexts: list[str], cluster_id: int) -> dict | None:
    context_block = "\n".join(f"- {c}" for c in contexts[:NEAREST_TOKENS_K])
    prompt = (
        f"Given these {len(contexts)} token contexts all activating the same internal cluster "
        "in a language model, identify the single most precise semantic concept they share.\n"
        f"Contexts:\n{context_block}\n\n"
        'Respond with JSON only, no preamble:\n'
        '{"symbol": "SNAKE_CASE_NAME", "type": "one of [Entity|Relation|Property|Process|Logical|Numeric|Modal]", '
        '"gloss": "one sentence definition", "confidence": 0.0-1.0}'
    )

    parsed = client.chat_json(prompt, context_label=f"centroid_{cluster_id}")
    if parsed is None:
        return None
    required = {"symbol", "type", "gloss", "confidence"}
    missing_keys = required - set(parsed.keys())
    if missing_keys:
        logger.warning(f"Centroid {cluster_id}: response missing keys {missing_keys} — skipping. Got: {list(parsed.keys())}")
        return None
    if parsed.get("type") not in VALID_TYPES:
        logger.warning(f"Centroid {cluster_id}: invalid type '{parsed.get('type')}' — skipping")
        return None
    if not isinstance(parsed.get("confidence"), (int, float)):
        logger.warning(f"Centroid {cluster_id}: missing confidence — skipping")
        return None

    return {
        "id": f"P{cluster_id:04d}",
        "symbol": str(parsed["symbol"]).upper().replace(" ", "_"),
        "type": parsed["type"],
        "gloss": str(parsed["gloss"]),
        "confidence": float(parsed["confidence"]),
    }


def deduplicate(primitives: list[dict], centroids: np.ndarray) -> list[dict]:
    """Remove near-duplicate primitives (cosine sim > threshold), keep higher confidence."""
    ids = [p["id"] for p in primitives]
    id_to_idx = {p["id"]: i for i, p in enumerate(primitives)}
    centroid_idx = [int(p["id"][1:]) for p in primitives]
    vecs = centroids[centroid_idx].astype(np.float32)  # avoid float16 overflow in cosine sim

    sim = cosine_similarity_matrix(vecs, vecs)
    np.fill_diagonal(sim, 0.0)

    keep = set(range(len(primitives)))
    for i in range(len(primitives)):
        if i not in keep:
            continue
        for j in range(i + 1, len(primitives)):
            if j not in keep:
                continue
            if sim[i, j] > DEDUP_COSINE_THRESHOLD:
                loser = i if primitives[i]["confidence"] < primitives[j]["confidence"] else j
                keep.discard(loser)

    result = [primitives[i] for i in sorted(keep)]
    print(f"Deduplication: {len(primitives)} → {len(result)} primitives")
    return result


def main():
    logger.info("Phase 1 — label_primitives starting")
    logger.info("Loading centroids and metadata...")
    centroids = np.load(CENTROIDS_PATH)      # (k, d)
    assignments = np.load(ASSIGNMENTS_PATH)  # (N,)
    all_meta = load_meta(META_PATH)

    from config import ACTIVATIONS_PATH
    A = np.load(ACTIVATIONS_PATH)            # (N, d)

    client = get_client()

    primitives = []
    example_tokens_map = {}

    print(f"Labelling {len(centroids)} centroids...")
    for cluster_id in tqdm(range(len(centroids)), desc="Labelling"):
        centroid = centroids[cluster_id]  # (d,)

        member_indices = np.where(assignments == cluster_id)[0]
        if len(member_indices) == 0:
            continue

        member_vecs = A[member_indices]
        sims = member_vecs @ centroid / (
            np.linalg.norm(member_vecs, axis=1) * np.linalg.norm(centroid) + 1e-8
        )
        top_k_local = np.argsort(sims)[-NEAREST_TOKENS_K:][::-1]
        top_k_global = member_indices[top_k_local]

        contexts = []
        example_tokens = []
        for global_idx in top_k_global:
            m = all_meta[global_idx]
            contexts.append(build_context(m, all_meta))
            example_tokens.append(m["token_str"].strip())

        result = label_centroid(client, contexts, cluster_id)
        if result is None:
            continue
        if result["confidence"] < LABEL_MIN_CONFIDENCE:
            continue

        result["example_tokens"] = list(dict.fromkeys(example_tokens))[:10]
        result["centroid_embedding"] = centroids[cluster_id].tolist()
        primitives.append(result)

    logger.info(f"Retained {len(primitives)} primitives after confidence filter.")

    primitives = deduplicate(primitives, centroids)

    mean_conf = sum(p["confidence"] for p in primitives) / max(len(primitives), 1)
    logger.info(f"Mean confidence: {mean_conf:.3f}")
    if mean_conf < 0.65:
        logger.warning(
            "Mean confidence < 0.65. Consider re-running with a larger model or more corpus diversity."
        )

    with open(PRIMITIVES_PATH, "w", encoding="utf-8") as f:
        json.dump(primitives, f, indent=2)
    logger.info(f"Saved {len(primitives)} primitives to {PRIMITIVES_PATH}")
    mark_complete(1, f"{len(primitives)} primitives, mean_conf={mean_conf:.3f}")


if __name__ == "__main__":
    main()
