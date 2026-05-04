"""
Phase 4: Post-hoc corpus validation.

Re-validates every Neuralese chain in train.jsonl against the parser.
Reports any entries that have become invalid (e.g. after grammar changes).
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_CORPUS_PATH, CORPUS_STATS_PATH
from phase3_grammar.parser import validate, get_signatures


def main():
    if not TRAIN_CORPUS_PATH.exists():
        print(f"Corpus not found at {TRAIN_CORPUS_PATH}")
        sys.exit(1)

    sigs = get_signatures()
    valid_lines = []
    invalid_lines = []
    error_types: dict[str, int] = {}

    with open(TRAIN_CORPUS_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Re-validating {len(lines)} corpus entries...")
    for i, line in enumerate(tqdm(lines)):
        item = json.loads(line)
        chain = item.get("neuralese_chain", "")
        ok, err = validate(chain, sigs)
        if ok:
            valid_lines.append(line)
        else:
            invalid_lines.append((i, err, chain[:80]))
            err_type = err.split(":")[0] if err else "UNKNOWN"
            error_types[err_type] = error_types.get(err_type, 0) + 1

    print(f"\nValid:   {len(valid_lines)}/{len(lines)}")
    print(f"Invalid: {len(invalid_lines)}/{len(lines)}")

    if error_types:
        print("\nError type breakdown:")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {etype}: {count}")

    if invalid_lines:
        print("\nFirst 5 invalid entries:")
        for idx, err, chain_preview in invalid_lines[:5]:
            print(f"  [line {idx}] {err}")
            print(f"             {chain_preview}")

        cleaned_path = TRAIN_CORPUS_PATH.parent / "train_clean.jsonl"
        with open(cleaned_path, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)
        print(f"\nCleaned corpus written to {cleaned_path} ({len(valid_lines)} entries)")
    else:
        print("\nAll entries valid — corpus is clean.")

    stats_path = CORPUS_STATS_PATH
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        stats["_validation"] = {
            "total": len(lines),
            "valid": len(valid_lines),
            "invalid": len(invalid_lines),
            "error_types": error_types,
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
