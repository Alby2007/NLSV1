"""
Phase 5: Evaluation harness.

Runs the fine-tuned model on held-out eval splits, extracts
reasoning and answer blocks, scores them, and logs to outputs/eval_results.json.
"""

import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FINETUNE_MODEL, MAX_SEQ_LEN,
    CHECKPOINTS_DIR, EVAL_RESULTS_PATH,
    SOURCE_DATASETS,
)
from phase3_grammar.parser import validate, get_signatures

EVAL_SAMPLES_PER_DATASET = 200


def load_model(checkpoint_path: Path, device: str):
    # Load tokenizer from checkpoint — it has the custom Neuralese vocab injected during training
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    base = AutoModelForCausalLM.from_pretrained(
        FINETUNE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
    )
    # Resize to match custom vocab before loading LoRA weights
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, question: str, device: str, max_new_tokens: int = 256) -> str:
    prompt = f"<question> {question} </question>\n<reasoning>\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def extract_blocks(generated: str) -> tuple[str, str]:
    reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", generated, re.DOTALL)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", generated, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    return reasoning, answer


def score_answer(predicted: str, gold: str) -> bool:
    pred = predicted.strip().lower()
    g = gold.strip().lower()
    if pred == g:
        return True
    try:
        return abs(float(pred) - float(g)) < 1e-6
    except ValueError:
        pass
    return g in pred


def eval_dataset(tokenizer, model, dataset_cfg: dict, sigs, device: str) -> dict:
    name = dataset_cfg["name"]
    config = dataset_cfg.get("config")
    try:
        if config:
            ds = load_dataset(name, config, split="test")
        else:
            ds = load_dataset(name, split="test")
    except Exception:
        try:
            if config:
                ds = load_dataset(name, config, split="validation")
            else:
                ds = load_dataset(name, split="validation")
        except Exception as e:
            return {"error": str(e)}

    n = min(EVAL_SAMPLES_PER_DATASET, len(ds))
    ds = ds.shuffle(seed=42).select(range(n))

    correct = 0
    parse_valid = 0
    neuralese_lengths = []
    results = []

    for item in tqdm(ds, desc=f"  Eval {name}"):
        if name == "gsm8k":
            question = item["question"]
            gold_parts = item["answer"].split("####")
            gold = gold_parts[1].strip() if len(gold_parts) > 1 else item["answer"]
        elif "ai2_arc" in name:
            question = item["question"]
            ans_key = item["answerKey"]
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            idx = labels.index(ans_key) if ans_key in labels else 0
            gold = texts[idx]
        else:
            question = item.get("question", "")
            gold = str(item.get("answer", item.get("label", "")))

        generated = generate(tokenizer, model, question, device)
        reasoning, answer = extract_blocks(generated)

        ok, _ = validate(reasoning, sigs)
        if ok:
            parse_valid += 1

        is_correct = score_answer(answer, gold)
        if is_correct:
            correct += 1

        neuralese_lengths.append(len(reasoning.split()))
        results.append({
            "question": question,
            "gold": gold,
            "predicted_answer": answer,
            "correct": is_correct,
            "parse_valid": ok,
            "reasoning": reasoning[:200],
        })

    mean_len = sum(neuralese_lengths) / max(len(neuralese_lengths), 1)
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "parse_validity_rate": round(parse_valid / n, 4),
        "mean_chain_length": round(mean_len, 1),
        "sample_results": results[:20],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = CHECKPOINTS_DIR / "final"

    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Run train.py first.")
        sys.exit(1)

    print(f"Loading model from {checkpoint_path}")
    tokenizer, model = load_model(checkpoint_path, device)
    sigs = get_signatures()

    eval_results = {}
    for dataset_cfg in SOURCE_DATASETS:
        print(f"\nEvaluating {dataset_cfg['name']}...")
        result = eval_dataset(tokenizer, model, dataset_cfg, sigs, device)
        eval_results[dataset_cfg["name"]] = result
        if "accuracy" in result:
            print(f"  Accuracy:          {result['accuracy']:.1%}")
            print(f"  Parse validity:    {result['parse_validity_rate']:.1%}")
            print(f"  Mean chain length: {result['mean_chain_length']:.0f} tokens")

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEval results saved to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
