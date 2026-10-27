"""
Phase 5: LoRA fine-tuning on the Neuralese corpus.

Input:  data/corpus/train.jsonl
Output: outputs/checkpoints/
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

try:
    import bitsandbytes  # noqa: F401
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False

_REPO_ROOT = Path(__file__).parent.parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_REPO_ROOT / "outputs" / "train.log"),
    ],
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FINETUNE_MODEL,
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
    TRAIN_BATCH_SIZE, GRAD_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_EPOCHS, MAX_SEQ_LEN,
    EVAL_SPLIT_RATIO,
    TRAIN_CORPUS_PATH, CHECKPOINTS_DIR,
    CENTROIDS_PATH, ONTOLOGY_PATH,
)


# All legitimate ontology primitives that appear in the training corpus,
# ordered by corpus frequency. Derived from frequency analysis on 7,162 examples:
#   top-10 cover 93.7% of occurrences; all 55 cover 95% (remainder are hallucinations).
# These are expanded to atomic tokens so each symbol = 1 BPE token at inference time.
NEURALESE_TOKENS = [
    # Core arithmetic (>1000 occurrences each)
    "ARITH_EQUALS", "NUMERIC_RESULT", "ARITH_MULTIPLY", "ARITH_ADD",
    "ARITH_DIVIDE", "ARITH_SUBTRACT",
    # Causal / temporal (>50 occurrences)
    "CAUSAL_REQUIRES", "TEMPORAL_CHANGE", "CAUSAL_ENABLES", "TEMPORAL_DURING",
    "TEMPORAL_AFTER", "TEMPORAL_BEFORE", "CAUSAL_CONTRIBUTES", "CAUSAL_PREVENTS",
    # Accumulators / result markers
    "RUNNING_TOTAL", "ARITH_ASSIGN", "NUMERIC_VALUE", "RESULT",
    # Logical
    "LOGICAL_AND", "LOGICAL_NOT", "LOGICAL_IMPLIES", "THEREFORE",
    "FORALL", "EXISTS", "IS_EQUIVALENT_TO",
    # Comparators
    "ARITH_LESS_THAN", "ARITH_GREATER_THAN", "ARITH_MODULO",
    # Domain entities used in GSM8K (low freq but legitimate ontology symbols)
    "TOTAL", "EARNINGS", "HOURS", "REMAINING_AMOUNT", "DISTANCE",
    "COST", "TEMPORAL_REFERENCE", "WEEKS", "UNCERTAINTY_EXPRESSION",
    "MINUTES", "AMOUNT", "DAYS", "FEET", "DOLLARS", "REVENUE",
    "YEAR", "STUDENTS", "TOTAL_COUNT", "TOTAL_PAGES", "TOTAL_DAYS",
    "FINAL_RESULT", "TIME_PERIOD", "TIME_DURATION", "MILES", "POUNDS",
    # Aliases present in corpus (map to canonical ops — kept for backward compat)
    "DIVIDE", "EQUALS", "PLUS", "MULTIPLY",
    # Rare but valid
    "UNCERTAIN_EXPRESSION", "SHIRT",
    # Abstract quantity primitives (added to redirect hallucinated domain nouns)
    "NUMERIC_COUNT", "NUMERIC_RATE", "UNIT_COST", "NUMERIC_DURATION", "NUMERIC_QUANTITY",
]


PROMPT_TEMPLATE = (
    "<question> {question} </question>\n"
    "<reasoning>\n{neuralese_chain}\n</reasoning>\n"
    "<answer> {answer} </answer>"
)

PROMPT_ONLY_TEMPLATE = "<question> {question} </question>\n<reasoning>\n"


def load_corpus(corpus_path: Path) -> list[dict]:
    with open(corpus_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_example(item: dict, cot_field: str) -> str:
    chain = item[cot_field]
    if cot_field == "original_cot":
        # strip GSM8K calculator annotations <<400/20=20>>
        chain = re.sub(r"<<[^>]*>>", "", chain)
    return PROMPT_TEMPLATE.format(
        question=item["question"],
        neuralese_chain=chain,
        answer=item["answer"],
    )


def format_prompt(item: dict) -> str:
    return PROMPT_ONLY_TEMPLATE.format(question=item["question"])


def tokenize_fn(examples, tokenizer):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    prompts = tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    # mask the prompt — loss on reasoning chain + answer only
    enc["labels"] = [
        [-100] * min(len(p), len(ids)) + ids[min(len(p), len(ids)):]
        for ids, p in zip(enc["input_ids"], prompts["input_ids"])
    ]
    return enc


def main():
    parser = argparse.ArgumentParser(description="Phase 5 LoRA fine-tune")
    parser.add_argument("--load-mode", choices=["auto", "4bit", "fp16"], default="auto",
                        help="4bit = QLoRA via bitsandbytes (~5GB VRAM for 7B); "
                             "fp16 = full weights (~15GB for 7B — OOMs on 16GB cards)")
    parser.add_argument("--model", default=FINETUNE_MODEL,
                        help="override FINETUNE_MODEL (e.g. google/gemma-2-2b)")
    parser.add_argument("--corpus", type=Path, default=TRAIN_CORPUS_PATH,
                        help="training corpus jsonl (use train_final.jsonl for the roundtrip-filtered set)")
    parser.add_argument("--cot", choices=["neuralese_chain", "original_cot"],
                        default="neuralese_chain",
                        help="which field to train on: neuralese_chain = adapter B, "
                             "original_cot = adapter A (English CoT baseline)")
    parser.add_argument("--out", type=Path, default=CHECKPOINTS_DIR / "final",
                        help="output dir for the saved adapter (e.g. checkpoints/adapter_B_neuralese/final)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm also reports as cuda — correct
    model_name = args.model

    load_mode = args.load_mode
    if load_mode == "auto":
        if device == "cuda" and HAVE_BNB:
            load_mode = "4bit"
        elif device == "cuda":
            load_mode = "fp16"
            logging.warning(
                "bitsandbytes not found — loading fp16. A 7B model will likely OOM on 16GB; "
                "install bitsandbytes or pass --model google/gemma-2-2b"
            )
        else:
            load_mode = "fp16"
    if load_mode == "4bit" and not HAVE_BNB:
        raise RuntimeError("--load-mode 4bit requires bitsandbytes")
    logging.info(f"Device: {device} | load_mode: {load_mode} | model: {model_name}")

    logging.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Inject high-frequency Neuralese symbols as atomic tokens
    num_added = tokenizer.add_tokens(NEURALESE_TOKENS, special_tokens=False)
    logging.info(f"Added {num_added} Neuralese tokens to vocabulary (new vocab size: {len(tokenizer)})")

    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_kwargs = dict(
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
    )
    if load_mode == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if load_mode == "4bit":
        # casts norms to fp32 + enables non-reentrant checkpointing & input grads
        model = prepare_model_for_kbit_training(model)
    elif device == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if num_added > 0:
        # ── Build centroid lookup: symbol → centroid embedding ────────────────
        # Preferred: load Phase 1 activation centroids matched via ontology.
        # Fallback: subword mean (same as before). Either way we verify quality.
        import json
        centroid_map: dict[str, np.ndarray] = {}
        if CENTROIDS_PATH.exists() and ONTOLOGY_PATH.exists():
            logging.info(f"Loading centroid embeddings from {CENTROIDS_PATH}")
            centroids = np.load(str(CENTROIDS_PATH))  # shape: (K, d)
            ontology = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
            sym_to_centroid = {
                p["symbol"]: centroids[int(p["id"].lstrip("P"))]
                for p in ontology
                if "centroid_embedding" not in p  # use index-based lookup
                and p.get("id", "").startswith("P")
                and int(p["id"].lstrip("P")) < len(centroids)
            }
            # Also check for inline centroid_embedding field
            for p in ontology:
                if "centroid_embedding" in p and p["centroid_embedding"]:
                    sym_to_centroid[p["symbol"]] = np.array(p["centroid_embedding"], dtype=np.float32)
            centroid_map = sym_to_centroid
            logging.info(f"Centroid map built: {len(centroid_map)} symbols")
        else:
            logging.warning("centroids.npy or ontology.json not found — falling back to subword means")

        # Load original tokenizer for subword fallback
        orig_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        subword_ids_per_token = [orig_tok.encode(sym, add_special_tokens=False) for sym in NEURALESE_TOKENS]

        model.resize_token_embeddings(len(tokenizer))

        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            # Untied lm_head on Qwen — the output rows for new tokens get
            # HF's near-zero mean-resize init, which makes them near-impossible
            # to emit under frozen LoRA. Init them with the same vectors.
            out_emb = model.get_output_embeddings().weight
            same_head = out_emb.data_ptr() == emb.data_ptr()
            emb_dim = emb.shape[1]
            centroid_hits = 0
            subword_hits = 0

            for sym, new_id, subword_ids in zip(
                NEURALESE_TOKENS,
                tokenizer.convert_tokens_to_ids(NEURALESE_TOKENS),
                subword_ids_per_token,
            ):
                if sym in centroid_map:
                    # Use Phase 1 centroid — project/pad to model embedding dim if needed
                    c = torch.tensor(centroid_map[sym], dtype=emb.dtype)
                    if c.shape[0] != emb_dim:
                        # Centroid dim mismatch (e.g. Gemma centroids, Qwen model) — fall back
                        vec = emb[subword_ids].mean(dim=0)
                        subword_hits += 1
                    else:
                        vec = c
                        centroid_hits += 1
                else:
                    vec = emb[subword_ids].mean(dim=0)
                    subword_hits += 1
                emb[new_id] = vec
                if not same_head and new_id < out_emb.shape[0]:
                    out_emb[new_id] = vec.to(out_emb.dtype)

        logging.info(
            f"Embedding init: {centroid_hits} centroid, {subword_hits} subword-mean "
            f"(out of {len(NEURALESE_TOKENS)} tokens)"
        )

        # ── Cosine similarity sanity check ────────────────────────────────────
        # For each new token, verify its embedding is closer to its subword mean
        # than to a random token embedding (sanity, not a hard pass/fail).
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            rng = torch.Generator()
            rng.manual_seed(42)
            random_ids = torch.randint(0, orig_tok.vocab_size, (len(NEURALESE_TOKENS),), generator=rng)
            sims_to_subword, sims_to_random = [], []
            new_ids = tokenizer.convert_tokens_to_ids(NEURALESE_TOKENS)
            for new_id, subword_ids, rand_id in zip(new_ids, subword_ids_per_token, random_ids.tolist()):
                new_vec = emb[new_id].float()
                sw_mean = emb[subword_ids].float().mean(dim=0)
                rand_vec = emb[rand_id].float()
                sim_sw = F.cosine_similarity(new_vec.unsqueeze(0), sw_mean.unsqueeze(0)).item()
                sim_rand = F.cosine_similarity(new_vec.unsqueeze(0), rand_vec.unsqueeze(0)).item()
                sims_to_subword.append(sim_sw)
                sims_to_random.append(sim_rand)
            mean_sw = sum(sims_to_subword) / len(sims_to_subword)
            mean_rand = sum(sims_to_random) / len(sims_to_random)
            logging.info(
                f"Embedding sanity: mean cos-sim to subword-mean={mean_sw:.4f}, "
                f"to random token={mean_rand:.4f} "
                f"({'PASS' if mean_sw > mean_rand else 'WARN — embeddings may be misaligned'})"
            )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logging.info("LoRA applied.")

    corpus = load_corpus(args.corpus)
    logging.info(f"Corpus: {args.corpus} ({len(corpus)} examples, cot={args.cot})")

    texts = [format_example(item, args.cot) for item in corpus]
    prompts = [format_prompt(item) for item in corpus]
    dataset = Dataset.from_dict({"text": texts, "prompt": prompts})
    split = dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=42)
    map_kwargs = dict(batched=True, remove_columns=["text", "prompt"])
    train_ds = split["train"].map(lambda x: tokenize_fn(x, tokenizer), **map_kwargs)
    eval_ds = split["test"].map(lambda x: tokenize_fn(x, tokenizer), **map_kwargs)

    # Seq2Seq collator pads input_ids AND our pre-masked labels correctly
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=str(args.out.parent),   # step checkpoints sit next to final/
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=(device == "cuda"),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    logging.info("Starting training...")
    trainer.train()

    final_path = args.out
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))  # saves custom vocab with new tokens
    logging.info(f"Model saved to {final_path}")

    # Verify all Neuralese tokens are now atomic
    failures = []
    for sym in NEURALESE_TOKENS:
        toks = tokenizer.encode(sym, add_special_tokens=False)
        if len(toks) != 1:
            failures.append(f"{sym} -> {len(toks)} tokens")
    if failures:
        logging.warning(f"Token atomicity check FAILED: {failures}")
    else:
        logging.info(f"All {len(NEURALESE_TOKENS)} Neuralese tokens are atomic. Token ratio will improve at eval.")


if __name__ == "__main__":
    main()
