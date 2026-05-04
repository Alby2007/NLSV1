# Neuralese v1 — Project Specification
## A Human-Designed, LLM-Assisted Constructed Language for Model-Efficient Reasoning

---

## Project Goal

Build a compact, unambiguous, compositionally regular intermediate representation language — "Neuralese v1" — optimised for LLM reasoning efficiency. The target metric is: **reasoning chain length (tokens) to reach a correct answer**, minimised while maintaining accuracy.

This is not a natural language. It is closer to a typed IR (like LLVM IR) than to a conlang. Human-legible for debugging; optimised for model compute.

---

## Axis Decisions (Fixed for v1)

| Axis | Decision | Rationale |
|------|----------|-----------|
| Primitive representation | Discrete symbols, embeddings initialised to activation cluster centroids | Interpretable + semantically grounded |
| Structure | Hierarchical typed lambda calculus grammar | Compositional generalisation, type safety catches errors |
| Surface form | S-expressions (prefix, explicit arity) | Unambiguous, parse tree is surface form |
| Vocabulary | Static after mining (~2000 primitives) | Simplicity for v1 validation |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Primitive Mining                               │
│  Model activations → k-means clusters → labelled atoms  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Phase 2: Ontology Structuring                           │
│  Labelled atoms → LLM → typed hierarchy                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Phase 3: Grammar Generation                             │
│  Type hierarchy → LLM → context-free unambiguous grammar│
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Phase 4: Bootstrap Corpus                               │
│  CoT datasets → LLM translation → Neuralese corpus      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Phase 5: Fine-tune + Eval Loop                          │
│  Small model fine-tuned on corpus → eval → iterate      │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Primitive Mining

### Objective
Extract ~2000 candidate semantic primitives from a real model's internal representations.

### Model
Use `google/gemma-2-2b` via HuggingFace Transformers. Small enough to run on a single GPU/MPS/CPU, but deep enough to have rich middle-layer representations. If unavailable, fall back to `EleutherAI/pythia-1b`.

### Corpus
Sample 50,000 sentences from a mix of:
- `EleutherAI/pile` subset: DM Mathematics, FreeLaw, Wikipedia (reasoning diversity)
- `gsm8k` train split (arithmetic reasoning)
- `ai2_arc` train split (factual reasoning)
- `openwebmath` subset (formal reasoning)

Target: ~500 sentences per domain, diverse enough to activate broad concept space.

### Extraction procedure
1. Tokenise each sentence, run forward pass, extract residual stream at layer `num_layers // 2` (middle layer — past syntax, before output pressure).
2. Extract the activation vector for **every non-padding token position**.
3. Accumulate into a matrix A ∈ ℝ^(N × d) where N = total tokens across corpus, d = hidden dim.
4. Subsample to max 2,000,000 vectors if N exceeds this (random sample, preserve distribution).

### Clustering
Run **MiniBatch K-Means** with k=2048 on matrix A.
- Use `sklearn.cluster.MiniBatchKMeans`, batch_size=10000, n_init=5
- Save centroids C ∈ ℝ^(2048 × d) — these are candidate primitive embeddings
- Save cluster assignments for the full token set

### Primitive Labelling
For each centroid c_i:
1. Retrieve the 20 nearest tokens from the corpus (by cosine similarity of their activation vectors to c_i)
2. Collect the surrounding context window (±5 tokens) for each
3. Call Claude API with prompt:
   ```
   Given these 20 token contexts all activating the same internal cluster in a language model,
   identify the single most precise semantic concept they share.
   Respond with: {"symbol": "SNAKE_CASE_NAME", "type": "one of [Entity|Relation|Property|Process|Logical|Numeric|Modal]", "gloss": "one sentence definition", "confidence": 0.0-1.0}
   JSON only, no preamble.
   ```
4. Discard primitives with confidence < 0.6 (likely noisy clusters)
5. Deduplicate: if two primitives have cosine similarity > 0.92 between their centroid embeddings, keep the higher-confidence one

### Output
`primitives.json` — list of ~1500-2000 objects:
```json
{
  "id": "P0042",
  "symbol": "CAUSAL_ENABLES",
  "type": "Relation",
  "gloss": "X is a sufficient condition that brings about Y",
  "centroid_embedding": [...],
  "confidence": 0.87,
  "example_tokens": ["allows", "enables", "causes", "makes possible"]
}
```

---

## Phase 2: Ontology Structuring

### Objective
Organise the flat primitive list into a typed hierarchy that will ground the grammar's type system.

### Type Hierarchy (seed — LLM will refine)
```
Type
├── Entity
│   ├── ConcreteEntity
│   └── AbstractEntity
├── Relation
│   ├── CausalRelation
│   ├── TemporalRelation
│   ├── SpatialRelation
│   └── LogicalRelation
├── Property
│   ├── NumericProperty
│   └── QualitativeProperty
├── Process
│   ├── CognitiveProcess
│   └── PhysicalProcess
├── Logical
│   ├── Quantifier
│   ├── Connective
│   └── Comparator
├── Numeric
│   ├── Cardinal
│   ├── Ordinal
│   └── Operator
└── Modal
    ├── Epistemic  (belief, knowledge, uncertainty)
    └── Deontic    (obligation, permission)
```

### LLM Structuring Pass
Batch primitives by their assigned type. For each type bucket, call Claude API:
```
Here are N primitives of type {type}, each with symbol and gloss.
Assign each to the most specific subtype in this hierarchy: {subtypes}.
Also flag any primitive that belongs in a DIFFERENT top-level type than currently assigned.
Return JSON array: [{"id": "...", "subtype": "...", "reclassify_to": null | "NewType"}]
```

### Output
`ontology.json` — primitives with full type path:
```json
{
  "id": "P0042",
  "symbol": "CAUSAL_ENABLES",
  "type_path": ["Relation", "CausalRelation"],
  ...
}
```

---

## Phase 3: Grammar Generation

### Language Specification

#### Expressions
```
Expr    ::= Atom
          | (HEAD Expr*)          -- Application: head applied to args
          | (λ VAR:TYPE . Expr)   -- Abstraction: typed lambda
          | {Expr | CONF}         -- Assertion: proposition with confidence [0,1]
          | [Expr Expr]           -- Sequence: ordered pair / temporal chain
          | #VAR                  -- Variable reference

Atom    ::= primitive symbol (from ontology) | Number | #VAR

HEAD    ::= primitive symbol of type Relation | Process | Logical
CONF    ::= float literal 0.0–1.0
VAR     ::= lowercase identifier
TYPE    ::= type path leaf from ontology
Number  ::= integer | float
```

#### Type Rules (enforced by parser/validator)
- Application `(R A B)`: R must have arity matching arg count; arg types must match R's domain signature
- Abstraction `(λ x:T . E)`: x is bound in E, type T must exist in ontology
- Assertion `{P | c}`: P must be a well-typed Expr, c must be float in [0,1]
- Sequence `[A B]`: both A and B must be well-typed; result type is Sequence

#### Example expressions
```
; "Rain causes the ground to be wet"
(CAUSAL_PRODUCES RAIN_EVENT WET_GROUND_STATE)

; "If X is a prime number, then X has exactly two divisors"
(λ x:Numeric . (LOGICAL_IMPLIES
  {(IS_PRIME x) | 1.0}
  {(EQUALS (COUNT_DIVISORS x) 2) | 1.0}))

; "I believe with high confidence that Paris is the capital of France"
{(IS_CAPITAL PARIS FRANCE) | 0.92}

; Multi-step reasoning chain
[(CAUSAL_ENABLES STUDY_PROCESS KNOWLEDGE_GAIN)
 [(CAUSAL_ENABLES KNOWLEDGE_GAIN PROBLEM_SOLVING_ABILITY)
  {(THEREFORE PRACTICE_IMPROVES_PERFORMANCE) | 0.88}]]
```

### Grammar Validation
Write a Python parser (use `lark` library) that:
1. Parses any Neuralese expression against the grammar
2. Type-checks all applications against the ontology signature table
3. Returns either a parse tree or a typed error (ARITY_MISMATCH, TYPE_ERROR, UNKNOWN_SYMBOL)

The grammar is only accepted if the parser handles 100 hand-written test expressions correctly.

### LLM Grammar Assistance
After writing the initial grammar, call Claude API with the full primitive list and ask it to:
1. Identify any semantic patterns in the primitives that the grammar cannot express
2. Suggest grammar extensions (must not introduce ambiguity — validate each with the parser)
3. Propose the arity and type signature for every Relation and Process primitive

Output `grammar_signatures.json` — for every non-Atom primitive:
```json
{
  "symbol": "CAUSAL_ENABLES",
  "arity": 2,
  "domain": ["Process|Entity", "Process|Entity|Property"],
  "range": "Relation"
}
```

---

## Phase 4: Bootstrap Corpus

### Objective
Translate existing reasoning datasets into Neuralese to create fine-tuning data.

### Source Datasets
| Dataset | Split | N samples | Reasoning type |
|---------|-------|-----------|----------------|
| GSM8K | train | 7473 | Multi-step arithmetic |
| AI2-ARC (Challenge) | train | 1119 | Factual + causal |
| StrategyQA | train | 2290 | Compositional boolean |
| LogiQA | train | 7376 | Formal logic |

### Translation Pipeline
For each (question, chain-of-thought, answer) triple:

1. **Decompose** — call Claude API to extract the atomic reasoning steps from the CoT as a numbered list
2. **Map** — for each step, identify which Neuralese primitives are involved (fuzzy match against gloss definitions)
3. **Translate** — call Claude API with:
   ```
   Translate this reasoning step into Neuralese v1.
   Available primitives: {relevant_primitives_for_this_step}
   Grammar: {grammar_summary}
   Step: "{reasoning_step}"
   
   Rules:
   - Use only symbols from the provided primitive list
   - Every application must be well-typed per signatures
   - Wrap uncertain claims in {expr | confidence}
   - Prefer shorter expressions — this is a compression format
   - Return only the Neuralese expression, no explanation
   ```
4. **Validate** — run the parser on the output; if it fails, retry with error message appended (max 3 retries; discard if still failing)
5. **Assemble** — join validated step expressions into a Sequence chain

### Quality Filter
Discard any translated example where:
- Parser validation fails after 3 retries
- Neuralese token count > 1.5× original CoT token count (translation made it longer — defeat)
- Any UNKNOWN_SYMBOL error (primitive coverage gap)

### Output
`corpus/` directory:
- `train.jsonl` — one JSON per line: `{"question": "...", "neuralese_chain": "...", "answer": "...", "original_cot": "...", "compression_ratio": 0.xx}`
- `stats.json` — coverage rates, mean compression ratio, discard rate per dataset

Target: ≥10,000 valid translated examples before proceeding to Phase 5.

---

## Phase 5: Fine-tune + Eval Loop

### Model
`google/gemma-2-2b` (same as mining model — shared vocabulary base, consistent embedding space).
Fine-tune using LoRA (rank=16, alpha=32) via `peft` library. Train on consumer hardware target: single A100 40GB or equivalent.

### Training Objective
Standard causal language modelling on the neuralese corpus. Input: question in English. Output: Neuralese reasoning chain + answer.

Format:
```
<question> {English question text} </question>
<reasoning>
{Neuralese chain}
</reasoning>
<answer> {final answer} </answer>
```

### Evaluation
On held-out splits of each source dataset:

**Primary metric:** Answer accuracy (exact match or numeric equivalence)
**Secondary metrics:**
- Mean neuralese chain length vs mean CoT chain length (compression ratio)
- Parse validity rate (% of generated chains that parse correctly)
- Compositional generalisation: accuracy on concept combinations not seen in training

**Baseline:** Same model fine-tuned on English CoT (identical training setup, English reasoning chains instead of Neuralese)

### Eval Tooling
Write an eval harness that:
1. Runs the fine-tuned model on each eval question
2. Extracts the `<reasoning>` and `<answer>` blocks
3. Runs the parser on the reasoning block, logs validity
4. Scores the answer
5. Computes and logs all metrics to `eval_results.json`

### Iteration Signal
After each eval run, generate a failure report:
- Cluster failures by error type (parser error, wrong answer, etc.)
- For wrong-answer cases, trace back which step in the neuralese chain introduced the error
- Feed error patterns back to Phase 3 (grammar gaps) or Phase 1 (missing primitives)

---

## Repository Structure

```
neuralese/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters, paths, model names
│
├── phase1_mining/
│   ├── extract_activations.py # Forward passes, activation extraction
│   ├── cluster_primitives.py  # MiniBatch K-Means, centroid saving
│   └── label_primitives.py    # Claude API calls, primitive labelling
│
├── phase2_ontology/
│   └── structure_ontology.py  # LLM structuring pass, hierarchy assignment
│
├── phase3_grammar/
│   ├── grammar.lark            # Lark grammar definition
│   ├── parser.py              # Parser + type checker
│   ├── generate_signatures.py # LLM signature generation for primitives
│   └── test_grammar.py        # 100 test expressions + validation
│
├── phase4_corpus/
│   ├── translate_dataset.py   # Full translation pipeline
│   ├── validate_corpus.py     # Post-hoc corpus quality checks
│   └── corpus_stats.py        # Coverage, compression ratio reporting
│
├── phase5_finetune/
│   ├── train.py               # LoRA fine-tuning
│   ├── eval.py                # Eval harness
│   └── failure_report.py      # Error clustering + iteration signal
│
├── data/
│   ├── primitives.json         # Phase 1 output
│   ├── ontology.json           # Phase 2 output
│   ├── grammar_signatures.json # Phase 3 output
│   └── corpus/                 # Phase 4 output
│
└── outputs/
    ├── checkpoints/            # LoRA adapter weights
    └── eval_results.json       # Phase 5 output
```

---

## Requirements

```
torch>=2.1.0
transformers>=4.40.0
datasets>=2.18.0
scikit-learn>=1.4.0
lark>=1.1.9
peft>=0.10.0
anthropic>=0.25.0
numpy>=1.26.0
tqdm>=4.66.0
accelerate>=0.29.0
bitsandbytes>=0.43.0
```

---

## Key Design Constraints (Do Not Violate)

1. **Parser must be the source of truth.** Every expression in the corpus must pass the parser. No exceptions. Soft failures pollute the fine-tuning signal.

2. **Compression is the primary signal.** If a translation is longer than the original CoT, it is discarded. The language's entire value proposition is density.

3. **Confidence markers are mandatory for non-deductive steps.** Any step that is inductive, probabilistic, or uncertain must use `{expr | conf}`. Purely deductive steps (mathematical identities, logical tautologies) may omit them.

4. **No natural language inside reasoning chains.** The `<reasoning>` block must contain only valid Neuralese. English leakage defeats the evaluation.

5. **Phase 1 cluster quality gates the whole project.** If labelling confidence is low (mean < 0.65), re-run with a larger/better model or more corpus diversity before proceeding. Everything downstream depends on primitive quality.

---

## Success Criteria for v1

| Metric | Target |
|--------|--------|
| Primitive coverage (% of CoT concepts expressible) | ≥ 80% |
| Corpus size (valid translated examples) | ≥ 10,000 |
| Parse validity rate (generated chains) | ≥ 85% |
| Answer accuracy vs English CoT baseline | Within 5% |
| Mean compression ratio (neuralese vs CoT tokens) | ≤ 0.65 |

If compression ratio ≤ 0.65 and accuracy within 5% of baseline: v1 is a success. Proceed to v2 (dynamic vocabulary, continuous embeddings, multi-model communication protocol).

