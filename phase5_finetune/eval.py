"""
Phase 5: Evaluation harness — three-condition ablation.

Conditions:
  A  English CoT      — adapter trained on English chain-of-thought
  B  Neuralese        — adapter trained on Neuralese chains
  C  No reasoning     — base model, no fine-tuning, direct question→answer
  D  Zero-shot CoT    — base model, no fine-tuning, English CoT prompt

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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

try:
    import bitsandbytes  # noqa: F401
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FINETUNE_MODEL, MAX_SEQ_LEN,
    CHECKPOINTS_DIR, EVAL_RESULTS_PATH, OUTPUTS_DIR,
)
from phase3_grammar.parser import validate, validate_strict, get_signatures

INFERENCE_COMPRESSION_PATH = OUTPUTS_DIR / "inference_compression.json"
EVAL_SAMPLES_PER_DATASET = 200
MAX_NEW_TOKENS = 300
GEN_BATCH_SIZE = 8


def _load_kwargs(device: str) -> dict:
    """Model-load kwargs shared by all eval conditions — 8-bit when
    bitsandbytes is present. NOTE: 4-bit NF4 decode (seq_len=1 GEMV) is
    broken on gfx1103/ROCm — generates garbage despite correct full-seq
    forwards. 8-bit decode works and fits easily (~8GB)."""
    kwargs = dict(
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
    )
    if device == "cuda" and HAVE_BNB:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
    return kwargs

# ── Dataset configs ───────────────────────────────────────────────────────────
EVAL_DATASETS = [
    {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "domain": "in_distribution",
        "extract": "gsm8k",
    },
    {
        "name": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "domain": "ood",
        "extract": "math500",
    },
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_adapter_model(checkpoint_path: Path, device: str):
    """Load LoRA adapter (condition A or B) with its saved tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, **_load_kwargs(device))
    base.resize_token_embeddings(len(tokenizer))
    # The adapter doesn't carry the frozen embedding weights — resize just
    # mean+noise-initialized the new rows again. Rebuild the same
    # subword-mean init used at training time (input emb AND untied lm_head).
    from phase5_finetune.emb_init import init_neuralese_embeddings
    init_neuralese_embeddings(base, tokenizer)
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    model.eval()
    return tokenizer, model


def load_base_model(model_id: str, device: str):
    """Load raw base model for condition C (no fine-tuning)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_id, **_load_kwargs(device))
    model.eval()
    return tokenizer, model


# ── Generation ────────────────────────────────────────────────────────────────

def _generate_batch(tokenizer, model, prompts: list[str], max_new: int, device: str) -> list[tuple[str, int]]:
    """Batched greedy generation. Returns [(full_text, n_new_tokens)] per prompt."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    results = []
    for i in range(len(prompts)):
        generated_ids = out[i][input_len:]
        # Rows pad to the batch's longest generation — count real tokens only
        n_real = int((generated_ids != tokenizer.pad_token_id).sum())
        full_text = tokenizer.decode(out[i], skip_special_tokens=True)
        results.append((full_text, n_real))
    return results


def generate_with_reasoning(tokenizer, model, questions: list[str], device: str) -> list[tuple[str, int]]:
    """Generate with reasoning prompt (batched). Returns (full_text, n_reasoning_tokens) per q."""
    prompts = [f"<question> {q} </question>\n<reasoning>\n" for q in questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            # Stop at the answer close-tag — without this the model drifts
            # into hallucinated follow-up turns and every gen hits the cap.
            stop_strings=["</answer>"],
            tokenizer=tokenizer,
        )
    results = []
    for i in range(len(prompts)):
        generated_ids = out[i][input_len:]
        n_real = int((generated_ids != tokenizer.pad_token_id).sum())
        full_text = tokenizer.decode(out[i], skip_special_tokens=True)
        results.append((full_text, n_real))
    return results


def generate_direct(tokenizer, model, questions: list[str], device: str) -> list[tuple[str, int]]:
    """Condition C: direct question→answer with no reasoning scaffold."""
    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    return _generate_batch(tokenizer, model, prompts, 64, device)


def generate_cot_baseline(tokenizer, model, questions: list[str], device: str) -> list[tuple[str, int]]:
    """Condition D: base model zero-shot English CoT — the native-reasoning comparator."""
    prompts = [
        f"Question: {q}\nLet's think step by step, then give the final numeric answer."
        for q in questions
    ]
    return _generate_batch(tokenizer, model, prompts, MAX_NEW_TOKENS, device)


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


def extract_cot_answer(generated: str) -> str:
    """For condition D: last number mentioned in a free-form CoT output."""
    tail = generated.split("Question:")[-1]
    m = re.findall(r"answer is[:\s]*\(?\$?(-?[\d,]+\.?\d*)", tail, re.IGNORECASE)
    if m:
        return m[-1]
    nums = re.findall(r"-?[\d,]+\.?\d*", tail.replace(",", ""))
    return nums[-1] if nums else tail[:50].strip()


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
        elif extract == "math500":
            q = item["problem"]
            gold = str(item["answer"])
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
    parse_valid_strict = 0
    reasoning_token_counts = []
    results = []

    gen_fn = {
        "A": generate_with_reasoning,
        "B": generate_with_reasoning,
        "C": generate_direct,
        "D": generate_cot_baseline,
    }[condition]

    for start in tqdm(range(0, len(examples), GEN_BATCH_SIZE), desc=f"  [{condition}] {dataset_name}"):
        batch = examples[start:start + GEN_BATCH_SIZE]
        outs = gen_fn(tokenizer, model, [ex["question"] for ex in batch], device)

        for ex, (full_text, n_tokens) in zip(batch, outs):
            question = ex["question"]
            gold = ex["gold"]

            if condition == "C":
                answer = extract_direct_answer(full_text, question)
                reasoning = ""
            elif condition == "D":
                answer = extract_cot_answer(full_text)
                reasoning = full_text
            else:
                reasoning, answer = extract_blocks(full_text)

            reasoning_token_counts.append(n_tokens)

            is_correct = score_answer(answer, gold)
            if is_correct:
                correct += 1

            # Parse validity — only meaningful for condition B
            chain_valid = chain_valid_strict = False
            if condition == "B" and reasoning:
                chain_valid, _ = validate(reasoning, sigs)
                chain_valid_strict, _ = validate_strict(reasoning, sigs)
                parse_valid += chain_valid
                parse_valid_strict += chain_valid_strict

            results.append({
                "question": question[:120],
                "gold": gold,
                "predicted": answer,
                "correct": is_correct,
                "reasoning_tokens": n_tokens,
                "reasoning_preview": reasoning[:150],
                "parse_valid": chain_valid,
                "parse_valid_strict": chain_valid_strict,
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
        metrics["parse_validity_strict"] = round(parse_valid_strict / max(n, 1), 4)

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
    parser.add_argument("--conditions", type=str, default="A,B,C,D",
                        help="Comma-separated conditions to run (A, B, C, D).")
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

        # ── Conditions C + D on one base-model load ─────────────────────
        base_needed = [c for c in ("C", "D") if c in conditions_to_run]
        if base_needed:
            print(f"\nLoading base model (conditions {base_needed})...")
            tok_base, model_base = load_base_model(args.base_model, device)
            for cond in base_needed:
                label = "no reasoning" if cond == "C" else "zero-shot English CoT"
                metrics = eval_condition(cond, tok_base, model_base, examples, dataset_name, sigs, device)
                dataset_results[cond] = metrics
                print(f"  [{cond}] ({label}) accuracy={metrics['accuracy']:.1%}  "
                      f"tokens={metrics['mean_reasoning_tokens']:.0f}")
            del model_base
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
