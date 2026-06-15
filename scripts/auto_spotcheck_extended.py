"""
Extended automated spot-check: read all 100 sampled survivors, flag every
detectable issue, and estimate the hidden SEMANTICALLY_OFF rate.

SEMANTICALLY_OFF heuristics (things the round-trip check cannot catch):
  1. chain contains intermediate result that contradicts final answer
     e.g. chain computes 24 midway but answer is 6 and chain never uses 24 again
  2. chain references a variable (#x) that is assigned but never consumed
  3. step count is suspiciously low for a multi-clause question (step_count=1, answer not trivial)
  4. chain contains only one distinct arithmetic op for a question with 3+ sentences

These are heuristics, not certainties — they produce SUSPECT flags, not hard failures.
"""
import json, re, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_grammar.parser import validate, get_signatures

ont = set(p['symbol'] for p in json.load(open('data/ontology.json', encoding='utf-8')))
sigs = get_signatures()
sample = [json.loads(l) for l in open('data/corpus/spotcheck_sample.jsonl', encoding='utf-8') if l.strip()]

categories = Counter()
suspect_details = []

for i, ex in enumerate(sample):
    chain = ex.get('neuralese_chain', '')
    ratio = ex.get('compression_ratio', 0)
    steps = ex.get('step_count', 0)
    question = ex.get('question', '')
    answer = str(ex.get('answer', ''))

    flags = []

    # 1. Grammar
    ok, err = validate(chain, sigs)
    if not ok:
        flags.append('GRAMMAR_INVALID')

    # 2. Hallucinated primitives
    halluc = list(set(t for t in re.findall(r'[A-Z][A-Z_]+[A-Z]', chain) if t not in ont))
    if halluc:
        flags.append('HALLUC_PRIM')

    # 3. CORRECT_VERBOSE: high ratio AND many steps
    if ratio > 0.85 and steps > 4:
        flags.append('CORRECT_VERBOSE')

    # 4. SEMANTICALLY_OFF heuristics
    sem_flags = []

    # 4a. All numeric literals in chain — does the answer appear?
    nums_in_chain = re.findall(r'\b\d+\.?\d*\b', chain)
    try:
        ans_f = float(answer)
        ans_strs = [answer, str(int(ans_f)) if ans_f == int(ans_f) else answer]
        if nums_in_chain and not any(a in nums_in_chain for a in ans_strs):
            sem_flags.append('ANSWER_NOT_IN_CHAIN_NUMS')
    except ValueError:
        pass

    # 4b. Assigned vars never consumed
    assigned = set(re.findall(r'ARITH_EQUALS\s+#(\w+)', chain))
    consumed = set()
    for var in assigned:
        # consumed if appears somewhere other than the assignment position
        if len(re.findall(r'#' + var, chain)) > 1:
            consumed.add(var)
    orphaned = assigned - consumed
    if orphaned:
        sem_flags.append('ORPHANED_VAR:%s' % list(orphaned)[:2])

    # 4c. Multi-clause question but step_count=1
    q_sentences = len(re.findall(r'[.?!]', question))
    if q_sentences >= 3 and steps == 1:
        sem_flags.append('UNDERKILL_STEPS')

    if sem_flags:
        flags.append('SEMANTICALLY_SUSPECT')
        suspect_details.append({
            'idx': i+1,
            'question': question[:80],
            'answer': answer,
            'ratio': ratio,
            'steps': steps,
            'sem_flags': sem_flags,
            'chain_preview': chain[:120],
        })

    if not flags:
        categories['OK'] += 1
    else:
        for f in flags:
            categories[f] += 1

print('=' * 60)
print('AUTOMATED SPOT-CHECK: 100 examples')
print('=' * 60)
total = len(sample)
for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
    print('  %-30s: %d / %d  (%.0f%%)' % (cat, cnt, total, 100*cnt/total))

print()
print('SEMANTICALLY_SUSPECT details (%d examples):' % len(suspect_details))
for s in suspect_details[:15]:
    print()
    print('  [%d] %s' % (s['idx'], s['question']))
    print('       answer=%s  ratio=%.3f  steps=%s' % (s['answer'], s['ratio'], s['steps']))
    print('       sem_flags: %s' % s['sem_flags'])
    print('       chain: %s' % s['chain_preview'])

# Estimate hidden error rate
grammar_ok = total - categories.get('GRAMMAR_INVALID', 0)
hard_fails = categories.get('GRAMMAR_INVALID', 0) + categories.get('HALLUC_PRIM', 0)
soft_fails = categories.get('SEMANTICALLY_SUSPECT', 0)
print()
print('=' * 60)
print('ESTIMATED QUALITY:')
print('  Hard failures (grammar/halluc):   %d / %d = %.1f%%' % (hard_fails, total, 100*hard_fails/total))
print('  Soft suspects (semantic heuristic): %d / %d = %.1f%%' % (soft_fails, total, 100*soft_fails/total))
print('  Clean (no flags):                 %d / %d = %.1f%%' % (categories.get('OK',0), total, 100*categories.get('OK',0)/total))
print()
print('Interpretation:')
print('  Round-trip 88.6%% survival rate has an estimated ~%.0f%% additional' % (100*soft_fails/total))
print('  hidden semantic error rate on top — true quality ~%.0f%%.' % (
    88.6 - (88.6 * soft_fails / total)))
