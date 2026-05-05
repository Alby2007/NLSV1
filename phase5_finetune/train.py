"""
Phase 5: LoRA fine-tuning on the Neuralese corpus.

Input:  data/corpus/train.jsonl
Output: outputs/checkpoints/
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("outputs/train.log")],
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FINETUNE_MODEL,
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
    TRAIN_BATCH_SIZE, GRAD_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_EPOCHS, MAX_SEQ_LEN,
    EVAL_SPLIT_RATIO,
    TRAIN_CORPUS_PATH, CHECKPOINTS_DIR,
)


# High-frequency Neuralese primitives to add as atomic vocabulary tokens.
# Ordered by corpus frequency — top symbols fragment into 3-5 BPE pieces each.
NEURALESE_TOKENS = [
    "ARITH_EQUALS", "NUMERIC_RESULT", "ARITH_MULTIPLY", "ARITH_ADD",
    "ARITH_DIVIDE", "ARITH_SUBTRACT", "CAUSAL_REQUIRES", "TEMPORAL_CHANGE",
    "CAUSAL_ENABLES", "TEMPORAL_DURING", "TEMPORAL_AFTER", "TEMPORAL_BEFORE",
    "CAUSAL_CONTRIBUTES", "RUNNING_TOTAL", "LOGICAL_AND", "IS_EQUIVALENT_TO",
    "THEREFORE", "LOGICAL_NOT", "LOGICAL_IMPLIES", "FORALL", "EXISTS",
]


PROMPT_TEMPLATE = (
    "<question> {question} </question>\n"
    "<reasoning>\n{neuralese_chain}\n</reasoning>\n"
    "<answer> {answer} </answer>"
)


def load_corpus() -> list[dict]:
    with open(TRAIN_CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_example(item: dict) -> str:
    return PROMPT_TEMPLATE.format(
        question=item["question"],
        neuralese_chain=item["neuralese_chain"],
        answer=item["answer"],
    )


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    logging.info(f"Loading tokenizer and model: {FINETUNE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Inject high-frequency Neuralese symbols as atomic tokens
    num_added = tokenizer.add_tokens(NEURALESE_TOKENS, special_tokens=False)
    logging.info(f"Added {num_added} Neuralese tokens to vocabulary (new vocab size: {len(tokenizer)})")

    model = AutoModelForCausalLM.from_pretrained(
        FINETUNE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    if num_added > 0:
        # Load original tokenizer to get subword IDs before vocab expansion
        orig_tok = AutoTokenizer.from_pretrained(FINETUNE_MODEL, trust_remote_code=True)
        subword_ids_per_token = [orig_tok.encode(sym, add_special_tokens=False) for sym in NEURALESE_TOKENS]

        model.resize_token_embeddings(len(tokenizer))
        # Initialise new token embeddings to mean of their constituent subword pieces
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            for sym, new_id, subword_ids in zip(
                NEURALESE_TOKENS,
                tokenizer.convert_tokens_to_ids(NEURALESE_TOKENS),
                subword_ids_per_token,
            ):
                mean_vec = emb[subword_ids].mean(dim=0)
                emb[new_id] = mean_vec
        logging.info("New token embeddings initialised to subword means.")

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

    corpus = load_corpus()
    logging.info(f"Corpus size: {len(corpus)}")

    texts = [format_example(item) for item in corpus]
    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=EVAL_SPLIT_RATIO, seed=42)
    train_ds = split["train"].map(lambda x: tokenize_fn(x, tokenizer), batched=True, remove_columns=["text"])
    eval_ds = split["test"].map(lambda x: tokenize_fn(x, tokenizer), batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=(device == "cuda"),
        logging_steps=20,
        eval_strategy="no",
        save_strategy="no",
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

    final_path = CHECKPOINTS_DIR / "final"
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
