"""
Build the R3 retry list: GSM8K examples still not in the corpus after R2.
Saves to data/corpus/discarded_r3.jsonl
"""
import json
from pathlib import Path
from datasets import load_dataset

ds = load_dataset('gsm8k', 'main', split='train')
corpus = [json.loads(l) for l in Path('data/corpus/train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
corpus_questions = {e['question'].strip() for e in corpus}

def parse_gsm8k(ex):
    parts = ex['answer'].split('####')
    return ex['question'], parts[0].strip(), parts[1].strip() if len(parts) > 1 else ''

to_retry = []
for ex in ds:
    if ex['question'].strip() not in corpus_questions:
        q, cot, ans = parse_gsm8k(ex)
        to_retry.append({'question': q, 'cot': cot, 'answer': ans})

out = Path('data/corpus/discarded_r3.jsonl')
out.write_text('\n'.join(json.dumps(r) for r in to_retry), encoding='utf-8')

easy   = sum(1 for r in to_retry if len(r['cot'].split()) < 60)
medium = sum(1 for r in to_retry if 60 <= len(r['cot'].split()) < 120)
hard   = sum(1 for r in to_retry if len(r['cot'].split()) >= 120)
print(f'R3 retry list: {len(to_retry)} examples  (easy={easy} medium={medium} hard={hard})')
print(f'Current corpus: {len(corpus)}')
print(f'Saved to {out}')
