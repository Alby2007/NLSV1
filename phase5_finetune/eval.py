"""
Phase 5: Evaluation harness — three-condition ablation.

Conditions:
  A  English CoT      — adapter trained on English chain-of-thought
  B  Neuralese        — adapter trained on Neuralese chains
  C  No reasoning     — base model, no fine-tuning, direct question→answer

For each condition × dataset we measure:
  - Answer accuracy
  - Mean reasoning tokens generated at inference (greedy decoding)
  - Parse validity rate (Neuralese condition only)
  - Compression ratio at inference vs English CoT baseline

OOD test set: 200 examples from MATH (unseen domain during training).
In-distribution: 200 examples from GSM8K test split.

Results written to outputs/eval_results.json.
Inference compression written to outputs/inference_compression.json.

Usage:
    python phase5_finetune/eval.py \\
        --adapter_b outputs/checkpoints/adapter_B_neuralese/final \\
        --adapter_a outputs/checkpoints/adapter_A_english/final \\
        [--base_model google/gemma-2-2b]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FINETUNE_MODEL, MAX_SEQ_LEN,
    CHECKPOINTS_DIR, EVAL_RESULTS_PATH, OUTPUTS_DIR,
)
from phase3_grammar.parser import validate, get_signatures

INFERENCE_COMPRESSION_PATH = OUTPUTS_DIR / "inference_compression.json"
EVAL_SAMPLES_PER_DATASET = 200
MAX_NEW_TOKENS = 300

# ── Dataset configs ───────────────────────────────────────────────────────────
EVAL_DATASETS = [
    {
        "name": "gsm8k",
        "config": "main",
        "split": "test",
        "domain": "in_distribution",
        "extract": "gsm8k",
    },
    {
        "name": "lighteval/MATH",
        "config": "all",
        "split": "test",
        "domain": "ood",
        "extract": "math",
    },
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_adapter_model(checkpoint_path: Path, device: str):
    """Load LoRA adapter (condition A or B) with its saved tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        FINETUNE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    model.eval()
    return tokenizer, model


def load_base_model(model_id: str, device: str):
    """Load raw base model for condition C (no fine-tuning)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


# ── Generation ────────────────────────────────────────────────────────────────

def generate_with_reasoning(tokenizer, model, question: str, device: str) -> tuple[str, int]:
    """Generate with reasoning prompt. Returns (full_text, n_reasoning_tokens)."""
    prompt = f"<question> {question} </question>\n<reasoning>\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = out[0][input_len:]
    n_generated = len(generated_ids)
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return full_text, n_generated


def generate_direct(tokenizer, model, question: str, device: str) -> tuple[str, int]:
    """Condition C: direct question→answer with no reasoning scaffold."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,  # no reasoning, just the answer
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = out[0][input_len:]
    n_generated = len(generated_ids)
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return full_text, n_generated


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_blocks(generated: str) -> tuple[str, str]:
    reasoning_m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", generated, re.DOTALL)
    answer_m = re.search(r"<answer>\s*(.*?)\s*</answer>", generated, re.DOTALL)
    reasoning = reasoning_m.group(1).strip() if reasoning_m else ""
    answer = answer_m.group(1).strip() if answer_m else ""
    return reasoning, answer


def extract_direct_answer(generated: str, question: str) -> str:
    """For condition C: pull first number after 'Answer:' in the output."""
    after_answer = generated.split("Answer:")[-1].strip() if "Answer:" in generated else generated
    nums = re.findall(r"-?[\d,]+\.?\d*", after_answer.replace(",", ""))
    return nums[0] if nums else after_answer[:50].strip()


def normalize_answer(s: str) -> str:
    s = s.strip().lower().replace(",", "").replace("$", "").replace("%", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def score_answer(predicted: str, gold: str) -> bool:
    p, g = normalize_answer(predicted), normalize_answer(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-4
    except ValueError:
        return g in p or p in g


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_eval_examples(dataset_cfg: dict, n: int) -> list[dict]:
    """Returns list of {question, gold} dicts."""
    name = dataset_cfg["name"]
    config = dataset_cfg.get("config")
    split = dataset_cfg["split"]
    extract = dataset_cfg["extract"]

    examples = []
    try:
        ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    except Exception:
        try:
            ds = load_dataset(name, config, split="validation") if config else load_dataset(name, split="validation")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            return []

    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    for item in ds:
        if extract == "gsm8k":
            q = item["question"]
            parts = item["answer"].split("####")
            gold = parts[1].strip() if len(parts) > 1 else item["answer"]
        elif extract == "math":
            q = item.get("problem", item.get("question", ""))
            gold = item.get("solution", item.get("answer", ""))
            # MATH solutions are long; try to extract final numeric answer
            nums = re.findall(r"\\boxed\{([^}]+)\}", gold)
            gold = nums[-1] if nums else gold[:50]
        else:
            q = item.get("question", "")
            gold = str(item.get("answer", ""))
        examples.append({"question": q, "gold": gold})

    return examples


# ── Per-condition evaluation ───────────────────────────────────────────────────

def eval_condition(
    condition: str,
    tokenizer,
    model,
    examples: list[dict],
    dataset_name: str,
    sigs,
    device: str,
) -> dict:
    """Run one condition on one dataset. Returns metrics dict."""
    correct = 0
    parse_valid = 0
    reasoning_token_counts = []
    results = []

    for ex in tqdm(examples, desc=f"  [{condition}] {dataset_name}"):
        question = ex["question"]
        gold = ex["gold"]

        if condition == "C":
            full_text, n_tokens = generate_direct(tokenizer, model, question, device)
            answer = extract_direct_answer(full_text, question)
            reasoning = ""
        else:
            full_text, n_tokens = generate_with_reasoning(tokenizer, model, question, device)
            reasoning, answer = extract_blocks(full_text)

        reasoning_token_counts.append(n_tokens)

        is_correct = score_answer(answer, gold)
        if is_correct:
            correct += 1

        # Parse validity — only meaningful for condition B
        chain_valid = False
        if condition == "B" and reasoning:
            ok, _ = validate(reasoning, sigs)
            chain_valid = ok
            if ok:
                parse_valid += 1

        results.append({
            "question": question[:120],
            "gold": gold,
            "predicted": answer,
            "correct": is_correct,
            "reasoning_tokens": n_tokens,
            "reasoning_preview": reasoning[:150],
            "parse_valid": chain_valid,
        })

    n = len(examples)
    mean_tokens = sum(reasoning_token_counts) / max(n, 1)
    metrics = {
        "condition": condition,
        "dataset": dataset_name,
        "n": n,
        "accuracy": round(correct / max(n, 1), 4),
        "mean_reasoning_tokens": round(mean_tokens, 1),
        "sample_results": results[:20],
    }
    if condition == "B":
        metrics["parse_validity_rate"] = round(parse_valid / max(n, 1), 4)

    return metrics


# ── Inference compression ─────────────────────────────────────────────────────

def compute_inference_compression(
    results_by_condition: dict,
    dataset_name: str,
) -> dict:
    """
    Compare mean reasoning tokens of condition B (Neuralese) to condition A (English CoT).
    This is the real compression metric — not measured on training data.
    """
    a_results = results_by_condition.get("A", {})
    b_results = results_by_condition.get("B", {})

    a_tokens = a_results.get("mean_reasoning_tokens")
    b_tokens = b_results.get("mean_reasoning_tokens")

    if a_tokens is None or b_tokens is None or a_tokens == 0:
        return {"dataset": dataset_name, "inference_compression": None, "note": "missing data"}

    ratio = b_tokens / a_tokens
    return {
        "dataset": dataset_name,
        "condition_A_mean_tokens": a_tokens,
        "condition_B_mean_tokens": b_tokens,
        "inference_compression_ratio": round(ratio, 4),
        "compression_note": (
            f"Neuralese uses {ratio:.2%} of English CoT tokens at inference "
            f"({'compressed' if ratio < 1.0 else 'expanded'})"
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_b", type=str,
                        default=str(CHECKPOINTS_DIR / "adapter_B_neuralese" / "final"),
                        help="Path to Adapter B (Neuralese) checkpoint.")
    parser.add_argument("--adapter_a", type=str,
                        default=str(CHECKPOINTS_DIR / "adapter_A_english" / "final"),
                        help="Path to Adapter A (English CoT) checkpoint.")
    parser.add_argument("--base_model", type=str, default=FINETUNE_MODEL,
                        help="Base model ID for condition C (no fine-tuning).")
    parser.add_argument("--conditions", type=str, default="A,B,C",
                        help="Comma-separated conditions to run (A, B, C).")
    parser.add_argument("--n", type=int, default=EVAL_SAMPLES_PER_DATASET,
                        help="Examples per dataset.")
    args = parser.parse_args()

    conditions_to_run = [c.strip().upper() for c in args.conditions.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Conditions: {conditions_to_run}")

    sigs = get_signatures()
    all_results: dict = {}
    inference_compression: list[dict] = []

    for dataset_cfg in EVAL_DATASETS:
        dataset_name = dataset_cfg["name"].split("/")[-1]
        domain = dataset_cfg["domain"]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({domain})")
        print(f"{'='*60}")

        examples = load_eval_examples(dataset_cfg, args.n)
        if not examples:
            print(f"  Skipping {dataset_name} — failed to load.")
            continue
        print(f"  Loaded {len(examples)} examples.")

        dataset_results: dict[str, dict] = {}

        # ── Condition B: Neuralese ──────────────────────────────────────
        if "B" in conditions_to_run:
            b_path = Path(args.adapter_b)
            if not b_path.exists():
                print(f"  Adapter B not found at {b_path} — skipping condition B.")
            else:
                print(f"\nLoading Adapter B (Neuralese)...")
                tok_b, model_b = load_adapter_model(b_path, device)
                metrics_b = eval_condition("B", tok_b, model_b, examples, dataset_name, sigs, device)
                dataset_results["B"] = metrics_b
                print(f"  [B] accuracy={metrics_b['accuracy']:.1%}  "
                      f"tokens={metrics_b['mean_reasoning_tokens']:.0f}  "
                      f"parse_valid={metrics_b.get('parse_validity_rate', 'N/A')}")
                del model_b
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ── Condition A: English CoT ────────────────────────────────────
        if "A" in conditions_to_run:
            a_path = Path(args.adapter_a)
            if not a_path.exists():
                print(f"  Adapter A not found at {a_path} — skipping condition A.")
            else:
                print(f"\nLoading Adapter A (English CoT)...")
                tok_a, model_a = load_adapter_model(a_path, device)
                metrics_a = eval_condition("A", tok_a, model_a, examples, dataset_name, sigs, device)
                dataset_results["A"] = metrics_a
                print(f"  [A] accuracy={metrics_a['accuracy']:.1%}  "
                      f"tokens={metrics_a['mean_reasoning_tokens']:.0f}")
                del model_a
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ── Condition C: No reasoning (base model) ──────────────────────
        if "C" in conditions_to_run:
            print(f"\nLoading base model (Condition C: no reasoning)...")
            tok_c, model_c = load_base_model(args.base_model, device)
            metrics_c = eval_condition("C", tok_c, model_c, examples, dataset_name, sigs, device)
            dataset_results["C"] = metrics_c
            print(f"  [C] accuracy={metrics_c['accuracy']:.1%}  "
                  f"tokens={metrics_c['mean_reasoning_tokens']:.0f}")
            del model_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[dataset_name] = dataset_results

        # ── Inference compression ───────────────────────────────────────
        comp = compute_inference_compression(dataset_results, dataset_name)
        inference_compression.append(comp)
        if comp.get("inference_compression_ratio") is not None:
            print(f"\n  Inference compression: {comp['compression_note']}")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEval results saved to {EVAL_RESULTS_PATH}")

    with open(INFERENCE_COMPRESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(inference_compression, f, indent=2)
    print(f"Inference compression saved to {INFERENCE_COMPRESSION_PATH}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Cond':<6} {'Accuracy':>10} {'Tokens':>10} {'ParseValid':>12}")
    print("-" * 70)
    for ds_name, ds_results in all_results.items():
        for cond, metrics in sorted(ds_results.items()):
            acc = f"{metrics['accuracy']:.1%}"
            toks = f"{metrics['mean_reasoning_tokens']:.0f}"
            pv = f"{metrics.get('parse_validity_rate', ''):.1%}" if "parse_validity_rate" in metrics else "—"
            print(f"{ds_name:<15} {cond:<6} {acc:>10} {toks:>10} {pv:>12}")
    print("-" * 70)
    for comp in inference_compression:
        if comp.get("inference_compression_ratio") is not None:
            print(f"\n{comp['dataset']}: {comp['compression_note']}")


if __name__ == "__main__":
    main()
