# Phase 5 Pass/Fail Criteria

Written before running Phase 5 so results can be interpreted without post-hoc rationalization.

## Conditions

| Label | Description |
|---|---|
| A | LoRA adapter fine-tuned on English CoT (baseline) |
| B | LoRA adapter fine-tuned on Neuralese chains |
| C | Base model, no fine-tuning, direct question→answer (floor) |

## Datasets

- **GSM8K test** (in-distribution, 200 examples) — primary eval
- **MATH test** (OOD, 200 examples) — generalization signal

## Metrics Reported

1. **Answer accuracy** — exact match or numeric match within 1e-4
2. **Mean reasoning tokens at inference** — greedy decoding, measured on generated output
3. **Inference compression ratio** — B mean tokens / A mean tokens (NOT training corpus ratio)
4. **Parse validity rate** — condition B only: % of generated Neuralese chains that parse

---

## Pass / Fail Definitions

### Primary hypothesis: Neuralese compresses inference tokens while maintaining accuracy

**PASS (strong):**
- B accuracy within 5pp of A on GSM8K
- Inference compression ratio < 0.80 (Neuralese uses ≤80% of English CoT tokens)
- B accuracy > C accuracy (fine-tuning on Neuralese is better than no reasoning at all)

**PASS (weak):**
- B accuracy within 10pp of A on GSM8K
- Inference compression ratio < 0.95
- B accuracy > C accuracy

**AMBIGUOUS:**
- B accuracy within 5pp of A, but compression ratio ≥ 0.95 (Neuralese isn't actually shorter at inference)
- B accuracy within 5pp of A, but parse validity < 50% (model is generating plausible-looking but invalid Neuralese)

**FAIL:**
- B accuracy more than 10pp below A on GSM8K (Neuralese hurts reasoning quality too much to be worthwhile)
- B accuracy ≤ C accuracy (fine-tuning on Neuralese provides no reasoning benefit over the floor)
- Inference compression ratio > 1.0 (Neuralese is longer than English at inference — negative result)

---

## OOD Interpretation

MATH performance for all conditions will likely be much lower than GSM8K (it's harder and unseen).
The OOD signal to watch: **does the gap between A and B widen or narrow on MATH?**

- Gap narrows on MATH → Neuralese generalizes better (positive signal for symbolic representations)
- Gap widens on MATH → Neuralese is overfitting the training domain (expected, not fatal)
- B > A on MATH → unexpected, investigate before claiming

---

## What to do with the result

### If PASS (strong or weak):
- Document inference compression as the key finding
- Note parse validity rate as a secondary metric
- Plan domain diversity expansion (ARC, StrategyQA) for v1.1

### If AMBIGUOUS:
- Check whether the model generates any valid Neuralese at all (parse validity)
- If parse validity > 50%: the grammar is learned, accuracy may improve with more data or epochs
- If parse validity < 50%: the surface form is not being learned — reconsider training data quality

### If FAIL:
- Check training loss curves — did Adapter B converge similarly to Adapter A?
- Check if compression_ratio > 1.0 examples slipped through (should have been filtered)
- Check spot-check results for systematic translation quality issues
- Consider: is the Neuralese grammar learnable from 7K examples of GSM8K only?

---

## Notes

- These criteria were written on 2026-06-15, before Phase 5 training was run.
- The 5pp threshold is borrowed from standard NLP significance conventions; it is not derived from power analysis.
- Parse validity is a proxy, not a direct measure of reasoning quality.
