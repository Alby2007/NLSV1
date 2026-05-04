"""
Phase 1 — Step 1: Extract residual stream activations from the mining model.

Saves a memory-mapped numpy array of activation vectors to data/activations.npy
and a corresponding token metadata file to data/activation_meta.jsonl.
"""

import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MINING_MODEL, MINING_MODEL_FALLBACK,
    CORPUS_SAMPLE_SIZE, MAX_ACTIVATION_VECTORS,
    LAYER_FRACTION, ACTIVATION_DTYPE,
    ACTIVATIONS_PATH, ACTIVATION_META_PATH,
)
from utils.logging import get_logger

logger = get_logger(__name__)
META_PATH = ACTIVATION_META_PATH

_NP_DTYPE = np.float16 if ACTIVATION_DTYPE == "float16" else np.float32
_TORCH_DTYPE = torch.float16 if ACTIVATION_DTYPE == "float16" else torch.float32


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str):
    logger.info(f"Loading model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Do NOT pass output_hidden_states=True — we use a hook instead to grab
    # the residual stream output directly from model.layers[layer_idx].
    # hidden_states[i] in HF is offset by 1 (index 0 = embedding output),
    # so hidden_states[layer_idx] would give layer_idx-1's output. The hook
    # avoids this off-by-one and captures the correct residual stream tensor.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    model.eval()
    return tokenizer, model


def sample_corpus(n: int) -> list[str]:
    """Sample sentences from mixed reasoning corpora."""
    sentences = []
    per_domain = n // 4

    logger.info("Loading GSM8K...")
    gsm = load_dataset("gsm8k", "main", split="train")
    for item in gsm.shuffle(seed=42).select(range(min(per_domain, len(gsm)))):
        sentences.append(item["question"])

    logger.info("Loading ARC...")
    arc = load_dataset("ai2_arc", "ARC-Challenge", split="train")
    for item in arc.shuffle(seed=42).select(range(min(per_domain, len(arc)))):
        sentences.append(item["question"])

    logger.info("Loading Wikipedia (from pile-uncopyrighted)...")
    try:
        wiki = load_dataset("monology/pile-uncopyrighted", "default", split="train", streaming=True)
        count = 0
        for item in wiki:
            if item.get("meta", {}).get("pile_set_name") == "Wikipedia (en)":
                sentences.append(item["text"][:512])
                count += 1
                if count >= per_domain:
                    break
    except Exception as e:
        logger.warning(f"Wikipedia load failed ({e}), skipping.")

    logger.info("Loading OpenWebMath...")
    try:
        owm = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        count = 0
        for item in owm:
            sentences.append(item["text"][:512])
            count += 1
            if count >= per_domain:
                break
    except Exception as e:
        logger.warning(f"OpenWebMath load failed ({e}), skipping.")

    random.shuffle(sentences)
    return sentences[:n]


def extract_activations(sentences: list[str], tokenizer, model, device: str, layer_idx: int):
    """
    Run forward passes and collect residual stream activations via a hook.

    Uses register_forward_hook on model.model.layers[layer_idx] so we capture
    the output of that transformer block directly — the true residual stream
    after the layer norm — rather than relying on hidden_states indexing which
    is off by one due to the embedding layer prepended at index 0.
    """
    all_vectors = []
    all_meta = []
    _hook_output: list = []

    # Reach into the model's transformer block list.
    # Different architectures use different attribute paths:
    #   model.model.layers      — Gemma-2, Llama, Mistral, Qwen2
    #   model.gpt_neox.layers   — Pythia, GPT-NeoX
    #   model.transformer.h     — GPT-2 style
    def _get_layers(m):
        for attr_path in ("model.layers", "gpt_neox.layers", "transformer.h"):
            obj = m
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError(
            f"Cannot locate transformer layer list on {type(m).__name__}. "
            "Add its attribute path to _get_layers()."
        )

    transformer_layers = _get_layers(model)
    if layer_idx >= len(transformer_layers):
        raise ValueError(
            f"layer_idx={layer_idx} out of range for model with {len(transformer_layers)} layers"
        )

    def _hook(module, input, output):
        # Gemma-2 layer output is a tuple: (hidden_states, optional_kv_cache, ...)
        # Always unpack index 0 — grabbing the tuple directly would silently
        # store the wrong object and corrupt every downstream vector.
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden shape: (batch=1, seq_len, d_model)
        # Convert to float32 on CPU before numpy() — casting a CUDA float16
        # tensor directly to numpy requires a contiguous CPU copy; doing it
        # in one step avoids silent precision/layout issues. Then cast to
        # _NP_DTYPE for storage efficiency.
        arr = hidden.detach().float().cpu().numpy().astype(_NP_DTYPE)
        _hook_output.append(arr)  # shape: (1, seq_len, d_model), dtype=_NP_DTYPE

    handle = transformer_layers[layer_idx].register_forward_hook(_hook)

    try:
        for sent_idx, sentence in enumerate(tqdm(sentences, desc="Extracting activations")):
            _hook_output.clear()

            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False,
            ).to(device)

            with torch.no_grad():
                model(**inputs)

            if not _hook_output:
                logger.warning(f"No hook output for sentence {sent_idx}, skipping.")
                continue

            hidden = _hook_output[0][0]  # (1, seq_len, d) → (seq_len, d), already numpy _NP_DTYPE
            input_ids = inputs["input_ids"].squeeze(0).tolist()
            attention_mask = inputs["attention_mask"].squeeze(0).tolist()

            for pos, (vec, tok_id, mask) in enumerate(zip(hidden, input_ids, attention_mask)):
                if mask == 0:
                    continue
                all_vectors.append(vec)  # already _NP_DTYPE from hook
                all_meta.append({
                    "sent_idx": sent_idx,
                    "pos": pos,
                    "token_id": tok_id,
                    "token_str": tokenizer.decode([tok_id]),
                    "sentence": sentence[:100],
                })

            if len(all_vectors) >= MAX_ACTIVATION_VECTORS:
                logger.info(f"Hit max vectors cap ({MAX_ACTIVATION_VECTORS}), stopping early.")
                break
    finally:
        handle.remove()

    return all_vectors, all_meta


def subsample(vectors: list, meta: list, max_n: int):
    if len(vectors) <= max_n:
        return vectors, meta
    idx = sorted(random.sample(range(len(vectors)), max_n))
    return [vectors[i] for i in idx], [meta[i] for i in idx]


if __name__ == "__main__":
    # ── Windows multiprocessing guard ─────────────────────────────────────────
    # On Windows, torch DataLoader (and HF datasets) spawn child processes.
    # Without this guard the module re-imports on fork, causing infinite loops.
    # All runtime code MUST live inside this block on Windows.

    logger.info("Phase 1 — extract_activations starting")
    device = get_device()
    logger.info(f"Device: {device}")

    try:
        tokenizer, model = load_model(MINING_MODEL, device)
    except Exception as e:
        logger.warning(f"Primary model failed ({e}), falling back to {MINING_MODEL_FALLBACK}")
        tokenizer, model = load_model(MINING_MODEL_FALLBACK, device)

    num_layers = model.config.num_hidden_layers
    layer_idx = int(num_layers * LAYER_FRACTION)
    logger.info(
        f"Model has {num_layers} layers. "
        f"Extracting residual stream from layer {layer_idx} "
        f"(LAYER_FRACTION={LAYER_FRACTION}). "
        f"dtype={ACTIVATION_DTYPE}"
    )

    sentences = sample_corpus(CORPUS_SAMPLE_SIZE)
    logger.info(f"Sampled {len(sentences)} sentences.")

    vectors, meta = extract_activations(sentences, tokenizer, model, device, layer_idx)
    logger.info(f"Extracted {len(vectors)} activation vectors.")

    vectors, meta = subsample(vectors, meta, MAX_ACTIVATION_VECTORS)
    logger.info(f"After subsampling: {len(vectors)} vectors.")

    A = np.array(vectors, dtype=_NP_DTYPE)
    np.save(ACTIVATIONS_PATH, A)
    logger.info(f"Saved activations to {ACTIVATIONS_PATH}  shape={A.shape}  dtype={A.dtype}  "
                f"size={A.nbytes / 1e9:.2f} GB")

    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    logger.info(f"Saved metadata to {META_PATH}")
