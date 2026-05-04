"""
Phase 5: LoRA fine-tuning on the Neuralese corpus.

Input:  data/corpus/train.jsonl
Output: outputs/checkpoints/
"""

import json
import sys
from pathlib import Path
from typing import Optional

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

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FINETUNE_MODEL,
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
    TRAIN_BATCH_SIZE, GRAD_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_EPOCHS, MAX_SEQ_LEN,
    EVAL_SPLIT_RATIO,
    TRAIN_CORPUS_PATH, CHECKPOINTS_DIR,
)


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
    print(f"Device: {device}")

    print(f"Loading tokenizer and model: {FINETUNE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        FINETUNE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
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

    corpus = load_corpus()
    print(f"Corpus size: {len(corpus)}")

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
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    final_path = CHECKPOINTS_DIR / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
