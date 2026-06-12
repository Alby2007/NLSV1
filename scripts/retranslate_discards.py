"""
Re-translate the 2,220 previously discarded/filtered GSM8K examples
through the patched pipeline (Fix 1+2+3 applied).

Usage:
    python scripts/retranslate_discards.py

Appends new valid examples to data/corpus/train.jsonl
Writes failure log to outputs/retranslate_discards.log
"""
import json
import logging
import time
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ollama_client import get_client
from phase3_grammar.parser import get_signatures
from phase4_corpus.translate_dataset import translate_example

# ── Config ────────────────────────────────────────────────────────────────
DISCARDS_PATH   = Path('data/corpus/discarded_r3.jsonl')
CORPUS_PATH     = Path('data/corpus/train.jsonl')
LOG_PATH        = Path('outputs/retranslate_r3.log')
CHECKPOINT_PATH = Path('data/corpus/retranslate_r3_checkpoint.json')
OLLAMA_MODEL    = 'qwen2.5:14b'
CHECKPOINT_EVERY = 50

# ── Logging ───────────────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding='utf-8'))
    return {'completed': 0, 'recovered': 0}


def save_checkpoint(state):
    CHECKPOINT_PATH.write_text(json.dumps(state), encoding='utf-8')


def main():
    # Load inputs
    discards = [json.loads(l) for l in DISCARDS_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
    log.info(f'Loaded {len(discards)} examples to retry')

    existing = [json.loads(l) for l in CORPUS_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
    existing_questions = {e['question'].strip() for e in existing}
    log.info(f'Existing corpus: {len(existing)} examples')

    primitives = json.loads(Path('data/ontology.json').read_text(encoding='utf-8'))
    sigs = get_signatures()  # dict keyed by symbol name
    client = get_client(model=OLLAMA_MODEL)

    checkpoint = load_checkpoint()
    start_idx = checkpoint['completed']
    recovered_total = checkpoint['recovered']

    if start_idx > 0:
        log.info(f'Resuming from checkpoint at index {start_idx}')

    failure_reasons = Counter()
    new_examples = []
    t0 = time.time()

    for i, ex in enumerate(discards[start_idx:], start=start_idx):
        q = ex['question'].strip()
        if q in existing_questions:
            checkpoint['completed'] = i + 1
            continue

        result = translate_example(
            client=client,
            question=ex['question'],
            cot=ex['cot'],
            answer=ex['answer'],
            primitives=primitives,
            sigs=sigs,
        )

        if result is not None:
            new_examples.append(result)
            existing_questions.add(q)
            recovered_total += 1
            log.info(f'[{i+1}/{len(discards)}] PASS  ratio={result["compression_ratio"]:.3f}  recovered={recovered_total}')
        else:
            log.debug(f'[{i+1}/{len(discards)}] FAIL  q={ex["question"][:60]}')

        checkpoint['completed'] = i + 1
        checkpoint['recovered'] = recovered_total

        # Checkpoint + append to corpus periodically
        if len(new_examples) > 0 and (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
                for ex_out in new_examples:
                    f.write(json.dumps(ex_out) + '\n')
            log.info(f'Checkpoint: appended {len(new_examples)} examples. Total corpus now ~{len(existing)+recovered_total}')
            new_examples = []
            save_checkpoint(checkpoint)

        # ETA every 100 examples
        done = i + 1 - start_idx
        if done > 0 and done % 100 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = len(discards) - (i + 1)
            eta_min = remaining / rate / 60
            log.info(f'Progress: {i+1}/{len(discards)}  rate={rate:.2f}/s  ETA={eta_min:.0f}min  recovered={recovered_total}')

    # Final flush
    if new_examples:
        with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
            for ex_out in new_examples:
                f.write(json.dumps(ex_out) + '\n')
        save_checkpoint(checkpoint)

    # ── Final report ──────────────────────────────────────────────────────
    final_corpus = [json.loads(l) for l in CORPUS_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]
    recovered_examples = [e for e in final_corpus if e['question'].strip() not in
                          {e2['question'].strip() for e2 in existing}]

    log.info('=' * 60)
    log.info(f'RETRANSLATION COMPLETE')
    log.info(f'  Input:     {len(discards)} examples')
    log.info(f'  Recovered: {recovered_total}  ({100*recovered_total/max(len(discards),1):.1f}%)')
    log.info(f'  Still failing: {len(discards) - recovered_total}')
    log.info(f'  Final corpus size: {len(final_corpus)}')

    if recovered_total > 0:
        ratios = [e['compression_ratio'] for e in recovered_examples[:recovered_total]]
        mean_r = sum(ratios) / len(ratios)
        std_r  = (sum((r - mean_r)**2 for r in ratios) / len(ratios)) ** 0.5
        easy   = sum(1 for e in recovered_examples[:recovered_total] if len(e['original_cot'].split()) < 60)
        medium = sum(1 for e in recovered_examples[:recovered_total] if 60 <= len(e['original_cot'].split()) < 120)
        hard   = sum(1 for e in recovered_examples[:recovered_total] if len(e['original_cot'].split()) >= 120)
        log.info(f'  Recovered compression: mean={mean_r:.3f}  std={std_r:.3f}')
        log.info(f'  Difficulty split: easy={easy}  medium={medium}  hard={hard}')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
