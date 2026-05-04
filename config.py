import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

DATA_DIR.mkdir(exist_ok=True)
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

ACTIVATIONS_PATH = DATA_DIR / "activations.npy"
CENTROIDS_PATH = DATA_DIR / "centroids.npy"
ASSIGNMENTS_PATH = DATA_DIR / "assignments.npy"
ACTIVATION_META_PATH = DATA_DIR / "activation_meta.jsonl"
PRIMITIVES_PATH = DATA_DIR / "primitives.json"
ONTOLOGY_PATH = DATA_DIR / "ontology.json"
GRAMMAR_SIGNATURES_PATH = DATA_DIR / "grammar_signatures.json"
TRAIN_CORPUS_PATH = CORPUS_DIR / "train.jsonl"
CORPUS_STATS_PATH = CORPUS_DIR / "stats.json"
EVAL_RESULTS_PATH = OUTPUTS_DIR / "eval_results.json"

# ── Phase completion flags ─────────────────────────────────────────────────────
PHASE1_FLAG = DATA_DIR / ".phase1_complete"
PHASE2_FLAG = DATA_DIR / ".phase2_complete"
PHASE3_FLAG = DATA_DIR / ".phase3_complete"
PHASE4_FLAG = DATA_DIR / ".phase4_complete"

# ── Phase 1: Primitive Mining ──────────────────────────────────────────────────
MINING_MODEL = "google/gemma-2-2b"
MINING_MODEL_FALLBACK = "EleutherAI/pythia-1b"
CORPUS_SAMPLE_SIZE = 50_000
MAX_ACTIVATION_VECTORS = 500_000   # 500k × float16 × d=2304 ≈ 2.1 GB — manageable on most setups
KMEANS_K = 2048
KMEANS_BATCH_SIZE = 10_000
KMEANS_N_INIT = 5
LAYER_FRACTION = 0.5          # middle layer = num_layers * this; Gemma-2-2B has 26 layers → layer 13
ACTIVATION_DTYPE = "float16"  # halves file size vs float32, negligible quality loss for clustering
CONTEXT_WINDOW = 5            # ±tokens for labelling context
NEAREST_TOKENS_K = 20         # top-k tokens per centroid for labelling
LABEL_MIN_CONFIDENCE = 0.6
DEDUP_COSINE_THRESHOLD = 0.92

# ── Phase 2: Ontology ──────────────────────────────────────────────────────────
TYPE_HIERARCHY = {
    "Entity": ["ConcreteEntity", "AbstractEntity"],
    "Relation": ["CausalRelation", "TemporalRelation", "SpatialRelation", "LogicalRelation"],
    "Property": ["NumericProperty", "QualitativeProperty"],
    "Process": ["CognitiveProcess", "PhysicalProcess"],
    "Logical": ["Quantifier", "Connective", "Comparator"],
    "Numeric": ["Cardinal", "Ordinal", "Operator"],
    "Modal": ["Epistemic", "Deontic"],
}

# ── Phase 3: Grammar ──────────────────────────────────────────────────────────
GRAMMAR_PATH = ROOT / "phase3_grammar" / "grammar.lark"
GRAMMAR_TEST_COUNT = 100

# ── Phase 4: Corpus ───────────────────────────────────────────────────────────
TRANSLATION_MAX_RETRIES = 3
COMPRESSION_MAX_RATIO = 1.5   # discard if neuralese tokens > this × original
MIN_CORPUS_SIZE = 5_000   # GSM8K-only dry run; raise to 10_000 when all datasets enabled

SOURCE_DATASETS = [
    # GSM8K-first dry run — validate compression ratios and discard rate before full run.
    # Re-enable other datasets once GSM8K quality signal looks healthy (ratio ≤0.65, discard <30%).
    {"name": "gsm8k",             "split": "train", "n": 7473,  "type": "arithmetic", "config": "main"},
    # {"name": "ai2_arc",           "split": "train", "n": 1119,  "type": "factual",    "config": "ARC-Challenge"},
    # {"name": "strategyqa",        "split": "train", "n": 2290,  "type": "boolean"},
    # {"name": "lucasmccabe/logiqa","split": "train", "n": 7376,  "type": "logic"},
]

# ── Phase 5: Fine-tuning ──────────────────────────────────────────────────────
FINETUNE_MODEL = "google/gemma-2-2b"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
TRAIN_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LEN = 512
EVAL_SPLIT_RATIO = 0.1

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 300          # seconds per request — signature batches need headroom
