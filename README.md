# Neuralese v1

A human-designed, LLM-assisted constructed language for model-efficient reasoning.

## Goal

Build a compact, unambiguous, compositionally regular intermediate representation — optimised for LLM reasoning efficiency. Target metric: **reasoning chain length (tokens) to reach a correct answer**, minimised while maintaining accuracy.

## Pipeline

```
Phase 1  →  Primitive Mining         (gemma-2-2b activations → k-means → labelled atoms)
Phase 2  →  Ontology Structuring     (labelled atoms → typed hierarchy)
Phase 3  →  Grammar Generation       (type hierarchy → lark grammar + type checker)
Phase 4  →  Bootstrap Corpus         (CoT datasets → Neuralese translations)
Phase 5  →  Fine-tune + Eval Loop    (LoRA on gemma-2-2b → eval vs English CoT baseline)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) running locally with `qwen2.5:14b` pulled:

```bash
ollama pull qwen2.5:14b
```

## Running Each Phase

```bash
# Phase 1
python phase1_mining/extract_activations.py
python phase1_mining/cluster_primitives.py
python phase1_mining/label_primitives.py

# Phase 2
python phase2_ontology/structure_ontology.py

# Phase 3
python phase3_grammar/generate_signatures.py
python phase3_grammar/test_grammar.py   # must pass 100/100 before proceeding

# Phase 4
python phase4_corpus/translate_dataset.py
python phase4_corpus/validate_corpus.py
python phase4_corpus/corpus_stats.py

# Phase 5
python phase5_finetune/train.py
python phase5_finetune/eval.py
python phase5_finetune/failure_report.py
```

## Success Criteria (v1)

| Metric | Target |
|---|---|
| Primitive coverage | ≥ 80% |
| Valid corpus size | ≥ 10,000 examples |
| Parse validity rate | ≥ 85% |
| Answer accuracy vs CoT baseline | Within 5% |
| Mean compression ratio | ≤ 0.65 |

## Config

All hyperparameters and paths are in `config.py`.

## LLM Backend

Uses **Ollama + Qwen2.5 14B** locally for all labelling, structuring, and translation calls. No external API required. Configure `OLLAMA_MODEL` and `OLLAMA_HOST` in `config.py`.
