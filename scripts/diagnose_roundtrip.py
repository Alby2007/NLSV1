import json
from collections import Counter

fails = [json.loads(l) for l in open('data/corpus/roundtrip_failures.jsonl', encoding='utf-8') if l.strip()]
cats = Counter(f.get('failure_category','unknown') for f in fails)
print('Failure breakdown so far:')
for cat, count in cats.most_common():
    print('  %s: %d (%.1f%%)' % (cat, count, 100*count/max(len(fails),1)))
print('Total failures: %d' % len(fails))

mismatches = [f for f in fails if f.get('failure_category') == 'ANSWER_MISMATCH'][:3]
print()
print('Sample ANSWER_MISMATCH:')
for m in mismatches:
    print('  Q:', m['question'][:70])
    print('  gold=%s  predicted=%s' % (m['gold'], m['predicted']))
    print('  chain:', m['neuralese_chain'][:80])
    print()

parse_fails = [f for f in fails if f.get('failure_category') == 'PARSE_INVALID'][:2]
if parse_fails:
    print('Sample PARSE_INVALID:')
    for p in parse_fails:
        print('  chain:', p['neuralese_chain'][:80])
        print('  parse_error:', p.get('parse_error',''))
        print()
