"""Reconstruct Neuralese token embeddings at eval/load time.

LoRA adapters don't save the frozen base-model weights, so the embedding
rows for the 64 added tokens are re-created by resize_token_embeddings()
with HF's mean+noise init — which produced identical-norm output rows and
zero parseable generations (see pre-fix smoke test).

The training run initialized those rows as the mean of each symbol's
original subword embeddings (input AND untied lm_head). Those base rows are
frozen and deterministic, so re-running the same init here reproduces the
exact vectors the adapter trained against.
"""

import torch
from transformers import AutoTokenizer

from phase5_finetune.train import NEURALESE_TOKENS
from config import FINETUNE_MODEL


def init_neuralese_embeddings(model, tokenizer) -> None:
    """After resize_token_embeddings, re-init the new rows as subword-means.

    Applies to both input embeddings and the (untied) lm_head.
    """
    orig_tok = AutoTokenizer.from_pretrained(FINETUNE_MODEL, trust_remote_code=True)
    subword_ids_per_token = [
        orig_tok.encode(sym, add_special_tokens=False) for sym in NEURALESE_TOKENS
    ]
    new_ids = tokenizer.convert_tokens_to_ids(NEURALESE_TOKENS)

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        out_emb = model.get_output_embeddings().weight
        same_head = out_emb.data_ptr() == emb.data_ptr()
        for new_id, subword_ids in zip(new_ids, subword_ids_per_token):
            vec = emb[subword_ids].float().mean(dim=0)
            emb[new_id] = vec.to(emb.dtype)
            if not same_head and new_id < out_emb.shape[0]:
                out_emb[new_id] = vec.to(out_emb.dtype)
