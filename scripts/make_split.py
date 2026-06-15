"""
Build the final stratified train/test split from round-trip survivors.

Strategy:
  - Difficulty buckets: easy (<60 CoT words), medium (60-120), hard (>120)
  - easy + medium: 10% test
  - hard: oversample to ~40 test examples (or all hard if <40 total)
    Rationale: 12 hard examples at 10% is too thin for meaningful eval;
    40 gives ~5% significance at 80% power for a 10pp accuracy gap.
  - Random seed fixed for reproducibility.

Outputs:
  data/corpus/train_final.jsonl   — training set
  data/corpus/test_final.jsonl    — held-out test set
  outputs/split_stats.json        — split metadata

Also annotates every example with:
  difficulty: easy | medium | hard
  split:      train | test

Usage:
    python scripts/make_split.py [--survivors data/corpus/train_roundtrip.jsonl]
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS_DIR, OUTPUTS_DIR

TRAIN_FINAL_PATH = CORPUS_DIR / "train_final.jsonl"
TEST_FINAL_PATH  = CORPUS_DIR / "test_final.jsonl"
SPLIT_STATS_PATH = OUTPUTS_DIR / "split_stats.json"

HARD_TEST_TARGET = 40   # oversample hard bucket to this many test examples
EASY_MEDIUM_TEST_RATIO = 0.10
SEED = 42


def difficulty(ex: dict) -> str:
    w = len(ex.get("original_cot", "").split())
    if w < 60:  return "easy"
    if w < 120: return "medium"
    return "hard"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survivors", default=str(CORPUS_DIR / "train_roundtrip.jsonl"))
    args = parser.parse_args()

    survivors = [json.loads(l) for l in Path(args.survivors).read_text(encoding="utf-8").splitlines() if l.strip()]
    print("Loaded %d survivors." % len(survivors))

    # Annotate difficulty
    for ex in survivors:
        ex["difficulty"] = difficulty(ex)

    buckets: dict[str, list] = defaultdict(list)
    for ex in survivors:
        buckets[ex["difficulty"]].append(ex)

    rng = random.Random(SEED)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    train_set, test_set = [], []

    # easy + medium: 10% test
    for d in ("easy", "medium"):
        exs = buckets[d]
        n_test = max(1, int(len(exs) * EASY_MEDIUM_TEST_RATIO))
        test_set.extend(exs[:n_test])
        train_set.extend(exs[n_test:])
        print("  %s: total=%d  test=%d  train=%d" % (d, len(exs), n_test, len(exs)-n_test))

    # hard: oversample up to HARD_TEST_TARGET
    hard = buckets["hard"]
    n_hard_test = min(HARD_TEST_TARGET, len(hard))
    # If we have fewer hard examples than target, use all for test and warn
    if len(hard) < HARD_TEST_TARGET:
        print("  hard: only %d examples — using all %d for test (below target of %d)" % (
            len(hard), len(hard), HARD_TEST_TARGET))
        test_set.extend(hard)
        # no hard train examples in this case
    else:
        test_set.extend(hard[:n_hard_test])
        train_set.extend(hard[n_hard_test:])
        print("  hard: total=%d  test=%d  train=%d" % (len(hard), n_hard_test, len(hard)-n_hard_test))

    # Shuffle final sets
    rng.shuffle(train_set)
    rng.shuffle(test_set)

    # Annotate split field
    for ex in train_set: ex["split"] = "train"
    for ex in test_set:  ex["split"] = "test"

    # Write output
    TRAIN_FINAL_PATH.write_text(
        "\n".join(json.dumps(ex) for ex in train_set) + "\n", encoding="utf-8"
    )
    TEST_FINAL_PATH.write_text(
        "\n".join(json.dumps(ex) for ex in test_set) + "\n", encoding="utf-8"
    )

    # Stats
    test_diff = Counter(ex["difficulty"] for ex in test_set)
    train_diff = Counter(ex["difficulty"] for ex in train_set)
    stats = {
        "total_survivors": len(survivors),
        "train_n": len(train_set),
        "test_n": len(test_set),
        "train_by_difficulty": dict(train_diff),
        "test_by_difficulty": dict(test_diff),
        "seed": SEED,
        "hard_test_target": HARD_TEST_TARGET,
        "easy_medium_test_ratio": EASY_MEDIUM_TEST_RATIO,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print()
    print("=" * 50)
    print("SPLIT COMPLETE")
    print("  Train: %d  ->  %s" % (len(train_set), TRAIN_FINAL_PATH))
    print("  Test:  %d  ->  %s" % (len(test_set),  TEST_FINAL_PATH))
    print("  Test difficulty: easy=%d  medium=%d  hard=%d" % (
        test_diff.get("easy",0), test_diff.get("medium",0), test_diff.get("hard",0)))
    print("  Stats -> %s" % SPLIT_STATS_PATH)


if __name__ == "__main__":
    main()
