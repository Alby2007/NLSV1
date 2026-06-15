"""
Round-trip consistency check for the Neuralese corpus.

For each example in train.jsonl:
  1. Re-evaluate the Neuralese chain against the stored answer using Ollama.
     The model is asked: "Given this Neuralese reasoning chain, what is the answer?"
  2. Compare the extracted answer to the stored ground-truth answer.
  3. Flag examples where the chain produces a wrong or unparseable answer.

Survivors are written to data/corpus/train_roundtrip.jsonl.
Failures are written to data/corpus/roundtrip_failures.jsonl with a failure category.

Failure categories:
  PARSE_INVALID       — chain fails grammar validation (shouldn't exist post-Phase 4)
  ANSWER_MISMATCH     — chain re-evaluation gives wrong answer
  EVAL_FAIL           — Ollama call failed or returned no parseable number

Usage:
    python scripts/round_trip_check.py [--dry-run N]

--dry-run N: only process first N examples (for testing).
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRAIN_CORPUS_PATH, CORPUS_DIR
from phase3_grammar.parser import validate, get_signatures
from utils.ollama_client import get_client

SURVIVORS_PATH = CORPUS_DIR / "train_roundtrip.jsonl"
FAILURES_PATH = CORPUS_DIR / "roundtrip_failures.jsonl"
CHECKPOINT_PATH = CORPUS_DIR / "roundtrip_checkpoint.json"
LOG_PATH = Path("outputs/round_trip_check.log")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen2.5:14b"

EVAL_PROMPT_TEMPLATE = (
    "You are evaluating a Neuralese reasoning chain. "
    "Neuralese is a compact symbolic language where:\n"
    "  (ARITH_ADD a b) = a + b\n"
    "  (ARITH_MULTIPLY a b) = a × b\n"
    "  (ARITH_DIVIDE a b) = a ÷ b\n"
    "  (ARITH_SUBTRACT a b) = a - b\n"
    "  (ARITH_EQUALS x y) = x equals y\n"
    "  NUMERIC_RESULT = the final numeric answer\n"
    "  [A B] = do A then B in sequence\n\n"
    "Question: {question}\n\n"
    "Neuralese reasoning chain:\n{chain}\n\n"
    "What is the numeric answer? "
    "Reply with ONLY the number (e.g. 42 or 3.5). No explanation."
)


def normalize_answer(s: str) -> str:
    """Strip formatting, return canonical numeric string or lowercased text."""
    s = s.strip().lower().replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\s+", " ", s)
    # Remove trailing .0 for integers
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def answers_match(predicted: str, gold: str) -> bool:
    p = normalize_answer(predicted)
    g = normalize_answer(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-4
    except ValueError:
        return g in p or p in g


def extract_number(text: str) -> str | None:
    """Pull the first plausible numeric value out of a model response."""
    text = text.strip()
    # Direct number match
    m = re.match(r"^-?[\d,]+\.?\d*$", text)
    if m:
        return text.replace(",", "")
    # Number anywhere in the string
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if nums:
        return nums[0].replace(",", "")
    return None


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {"processed": 0, "survivors": 0, "failures": 0}


def save_checkpoint(state: dict):
    CHECKPOINT_PATH.write_text(json.dumps(state), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=int, default=None,
                        help="Only process first N examples.")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint (default: True).")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    # Load corpus
    raw_lines = [l for l in TRAIN_CORPUS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = []
    for l in raw_lines:
        try:
            corpus.append(json.loads(l))
        except json.JSONDecodeError:
            log.warning(f"Skipping malformed line: {l[:80]}")

    if args.dry_run:
        corpus = corpus[:args.dry_run]
        log.info(f"Dry run: processing first {args.dry_run} examples.")

    log.info(f"Loaded {len(corpus)} examples from corpus.")

    sigs = get_signatures()
    client = get_client(model=OLLAMA_MODEL)

    # Resume from checkpoint
    checkpoint = load_checkpoint() if args.resume else {"processed": 0, "survivors": 0, "failures": 0}
    start_idx = checkpoint["processed"] if args.resume else 0
    if start_idx > 0:
        log.info(f"Resuming from index {start_idx} ({checkpoint['survivors']} survivors, {checkpoint['failures']} failures so far).")

    failure_counts: Counter = Counter()
    t0 = time.time()

    with open(SURVIVORS_PATH, "a", encoding="utf-8") as surv_f, \
         open(FAILURES_PATH, "a", encoding="utf-8") as fail_f:

        for i, ex in enumerate(corpus[start_idx:], start=start_idx):
            question = ex.get("question", "")
            chain = ex.get("neuralese_chain", "")
            gold = str(ex.get("answer", "")).strip()

            # ── Step 1: Grammar validation ────────────────────────────────
            ok, parse_err = validate(chain, sigs)
            if not ok:
                failure_counts["PARSE_INVALID"] += 1
                fail_record = {**ex, "failure_category": "PARSE_INVALID", "parse_error": parse_err}
                fail_f.write(json.dumps(fail_record) + "\n")
                fail_f.flush()
                checkpoint["processed"] = i + 1
                checkpoint["failures"] = checkpoint.get("failures", 0) + 1
                log.debug(f"[{i+1}] PARSE_INVALID: {parse_err}")
                continue

            # ── Step 2: Re-evaluate via Ollama ────────────────────────────
            prompt = EVAL_PROMPT_TEMPLATE.format(question=question, chain=chain)
            try:
                raw_response = client.chat_text(prompt, context_label="round_trip")
            except Exception as e:
                raw_response = None
                log.warning(f"[{i+1}] Ollama call failed: {e}")

            if raw_response is None:
                failure_counts["EVAL_FAIL"] += 1
                fail_record = {**ex, "failure_category": "EVAL_FAIL", "ollama_response": None}
                fail_f.write(json.dumps(fail_record) + "\n")
                fail_f.flush()
                checkpoint["processed"] = i + 1
                checkpoint["failures"] = checkpoint.get("failures", 0) + 1
                continue

            predicted = extract_number(raw_response) or raw_response.strip()[:50]

            # ── Step 3: Answer comparison ─────────────────────────────────
            if answers_match(predicted, gold):
                surv_f.write(json.dumps(ex) + "\n")
                surv_f.flush()
                checkpoint["survivors"] = checkpoint.get("survivors", 0) + 1
                log.debug(f"[{i+1}] PASS  gold={gold}  predicted={predicted}")
            else:
                failure_counts["ANSWER_MISMATCH"] += 1
                fail_record = {
                    **ex,
                    "failure_category": "ANSWER_MISMATCH",
                    "gold": gold,
                    "predicted": predicted,
                    "ollama_response": raw_response[:200],
                }
                fail_f.write(json.dumps(fail_record) + "\n")
                fail_f.flush()
                checkpoint["failures"] = checkpoint.get("failures", 0) + 1
                log.debug(f"[{i+1}] MISMATCH  gold={gold}  predicted={predicted}")

            checkpoint["processed"] = i + 1

            # Progress + checkpoint every 50 examples
            done = i + 1 - start_idx
            if done % 50 == 0:
                save_checkpoint(checkpoint)
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = len(corpus) - (i + 1)
                eta_min = remaining / max(rate, 0.001) / 60
                log.info(
                    f"[{i+1}/{len(corpus)}] survivors={checkpoint['survivors']}  "
                    f"failures={checkpoint['failures']}  "
                    f"rate={rate:.2f}/s  ETA={eta_min:.0f}min"
                )

    save_checkpoint(checkpoint)

    # ── Final report ──────────────────────────────────────────────────────
    total = checkpoint["processed"]
    survivors = checkpoint["survivors"]
    failures = checkpoint["failures"]

    log.info("=" * 60)
    log.info("ROUND-TRIP CHECK COMPLETE")
    log.info(f"  Processed:  {total}")
    log.info(f"  Survivors:  {survivors} ({100*survivors/max(total,1):.1f}%)")
    log.info(f"  Failures:   {failures} ({100*failures/max(total,1):.1f}%)")
    log.info("  Failure breakdown:")
    for cat, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        log.info(f"    {cat}: {count}")
    log.info(f"  Survivors written to: {SURVIVORS_PATH}")
    log.info(f"  Failures written to:  {FAILURES_PATH}")
    log.info("=" * 60)

    print(f"\nSurvivors: {survivors}/{total} ({100*survivors/max(total,1):.1f}%)")
    print(f"Failures:  {failures}/{total} ({100*failures/max(total,1):.1f}%)")
    print(f"Breakdown: {dict(failure_counts)}")
    print(f"Use {SURVIVORS_PATH} as your Phase 5 training corpus.")


if __name__ == "__main__":
    main()
