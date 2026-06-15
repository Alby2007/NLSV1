"""
Upload the final corpus splits to HuggingFace.

Uploads:
  data/corpus/train_final.jsonl  -> split='train'
  data/corpus/test_final.jsonl   -> split='test'

Repo: ALBYJ07/neuralese-v1-corpus (dataset)
"""
import json, sys, os
from pathlib import Path
from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).parent.parent))

HF_TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('HF_TOKEN', '')
REPO_ID  = 'ALBYJ07/neuralese-v1-corpus'

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]

train = load_jsonl('data/corpus/train_final.jsonl')
test  = load_jsonl('data/corpus/test_final.jsonl')
print('Train: %d  Test: %d' % (len(train), len(test)))

ds = DatasetDict({
    'train': Dataset.from_list(train),
    'test':  Dataset.from_list(test),
})
print('Columns:', ds['train'].column_names)

ds.push_to_hub(REPO_ID, token=HF_TOKEN, private=False)
print('Done: https://huggingface.co/datasets/%s' % REPO_ID)
