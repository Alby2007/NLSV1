"""
Phase 4: Translate CoT reasoning datasets into Neuralese v1.

Pipeline per example:
  1. Decompose CoT into atomic steps (Ollama)
  2. Map steps to relevant primitives (fuzzy gloss match)
  3. Translate each step to Neuralese (Ollama, with parser validation)
  4. Assemble into sequence chain
  5. Quality filter and save to corpus/train.jsonl
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TRANSLATION_MAX_RETRIES, COMPRESSION_MAX_RATIO,
    SOURCE_DATASETS, ONTOLOGY_PATH,
    TRAIN_CORPUS_PATH, CORPUS_STATS_PATH,
)
from lark.exceptions import LarkError
from phase3_grammar.parser import validate, get_signatures
from utils.ollama_client import get_client
from utils.logging import get_logger
from utils.phases import require_phase, mark_complete

logger = get_logger(__name__)


def load_primitives() -> list[dict]:
    with open(ONTOLOGY_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_primitive_index(primitives: list[dict]) -> dict[str, dict]:
    return {p["symbol"]: p for p in primitives}


# Seeded structural primitives always included in every prompt.
# These are the connective tissue; Qwen must not invent equivalents.
_ALWAYS_INCLUDE_PREFIXES = (
    "ARITH_", "CAUSAL_", "LOGICAL_", "TEMPORAL_", "IS_INSTANCE_OF",
    "IS_PART_OF", "IS_EQUIVALENT_TO", "IS_GREATER_THAN", "IS_LESS_THAN",
    "THEREFORE", "BECAUSE", "FORALL", "EXISTS", "KNOWN_TRUE", "BELIEVED_TRUE",
    "UNCERTAIN",
)


def fuzzy_match_primitives(step_text: str, primitives: list[dict], top_k: int = 10) -> list[dict]:
    """Always include seeded structural primitives; supplement with top_k fuzzy matches."""
    always = [p for p in primitives if any(p["symbol"].startswith(pfx) or p["symbol"] == pfx
                                           for pfx in _ALWAYS_INCLUDE_PREFIXES)]
    always_syms = {p["symbol"] for p in always}

    step_words = set(re.findall(r"\w+", step_text.lower()))
    scored = []
    for p in primitives:
        if p["symbol"] in always_syms:
            continue
        gloss_words = set(re.findall(r"\w+", p["gloss"].lower()))
        symbol_words = set(re.findall(r"\w+", p["symbol"].lower()))
        overlap = len(step_words & (gloss_words | symbol_words))
        if overlap > 0:
            scored.append((overlap, p))
    scored.sort(key=lambda x: -x[0])
    fuzzy = [p for _, p in scored[:top_k]]
    return always + fuzzy


def decompose_cot(client, question: str, cot: str) -> list[str]:
    """Break CoT into atomic numbered reasoning steps."""
    prompt = (
        f"Question: {question}\n\nChain of thought: {cot}\n\n"
        "Extract the atomic reasoning steps as a numbered list. "
        "Each step should be a single logical operation or inference. "
        "Return JSON only: {\"steps\": [\"step1\", \"step2\", ...]}"
    )
    parsed = client.chat_json(prompt, context_label="decompose_cot")
    if parsed and isinstance(parsed, dict):
        steps = parsed.get("steps", [])
        return [str(s) for s in steps if s]
    sentences = re.split(r"[.!?]\s+", cot.strip())
    return [s.strip() for s in sentences if s.strip()][:10]


def translate_step(
    client,
    step: str,
    relevant_primitives: list[dict],
    grammar_summary: str,
    error_context: str = "",
    prior_candidate: str = "",
) -> str | None:
    """Translate one reasoning step to Neuralese. Returns expression string or None."""
    prim_list = "\n".join(
        f"  {p['symbol']} ({p['type']}): {p['gloss']}"
        for p in relevant_primitives[:15]
    )

    if error_context and prior_candidate:
        # Retry prompt: structurally different from first attempt.
        # Showing the prior candidate + specific error gives Ollama
        # something concrete to fix rather than regenerating from scratch.
        prompt = (
            f"This Neuralese expression is invalid:\n  {prior_candidate}\n\n"
            f"Parser error: {error_context}\n\n"
            f"Original step to express: \"{step}\"\n\n"
            f"Available primitives:\n{prim_list}\n\n"
            "Fix the error. Key constraints:\n"
            "- ALL atoms must be UPPER_CASE_SYMBOLS from the list — no lowercase, no quoted strings\n"
            "- Arity must match exactly\n"
            "Return ONLY the corrected Neuralese expression, no explanation, no markdown."
        )
    else:
        prompt = (
            "Translate this reasoning step into Neuralese v1.\n\n"
            f"Available primitives (use ONLY these):\n{prim_list}\n\n"
            f"Grammar summary:\n{grammar_summary}\n\n"
            f"Step to translate: \"{step}\"\n\n"
            "STRICT RULES — violations cause parse failure:\n"
            "1. Atoms: UPPER_CASE_SYMBOL from the list above OR bare number (42, 3.14, -1).\n"
            "   FORBIDDEN: lowercase words, quoted strings, invented symbols.\n"
            "   NEVER invent symbol names. Every symbol MUST appear in the primitive list above.\n"
            "   For unknown quantities use #var or NUMERIC_VALUE — NEVER use X, LET, VARIABLE,\n"
            "   NUMBER, HAVE, HAS, EAT, GIVE, AGE, DURATION, YEARS, I, UNCERTAIN_EXPRESSION.\n"
            "2. Applications: (HEAD arg1 arg2 ...) — parentheses required around every call.\n"
            "   BAD:  ARITH_ADD 3 4       GOOD: (ARITH_ADD 3 4)\n"
            "3. Arity must match the primitive exactly.\n"
            "4. Confidence: {expr | 0.9} — only for uncertain claims.\n"
            "5. Sequence:   [expr1 expr2 ...]\n\n"
            "VARIABLE USAGE — #var must be ALL LOWERCASE after #:\n"
            "  BAD:  TOTAL  RESULT  #TOTAL  #RESULT  ← uppercase forbidden as vars\n"
            "  GOOD: #total  #result  #n  #cost  #rate\n\n"
            "  BAD:  (ARITH_EQUALS TOTAL (ARITH_ADD 25 15))\n"
            "  GOOD: (ARITH_EQUALS #total (ARITH_ADD 25 15))\n\n"
            "  BAD:  (ARITH_MULTIPLY RESULT 3)\n"
            "  GOOD: [(ARITH_ADD 25 15) (ARITH_MULTIPLY #result 3)]\n\n"
            "MULTIPLE EXPRESSIONS must be wrapped in [...], never written side-by-side:\n"
            "  BAD:  (ARITH_ADD 3 4) (ARITH_EQUALS #x 7)   ← two bare expressions, invalid\n"
            "  GOOD: [(ARITH_ADD 3 4) (ARITH_EQUALS #x 7)] ← wrapped in sequence\n\n"
            "  Use NUMERIC_RESULT for the final answer atom, RUNNING_TOTAL for accumulated sums.\n\n"
            "Return ONLY the Neuralese expression. No explanation, no markdown, no prose."
        )

    raw = client.chat_text(prompt, context_label="translate_step")
    if raw is None:
        return None
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()
    if not raw:
        return None
    # Auto-fix 1: lowercase #VAR references (Qwen often writes #TOTAL instead of #total)
    raw = re.sub(r"#([A-Z][A-Z0-9_]*)", lambda m: "#" + m.group(1).lower(), raw)
    # Auto-fix 2: structural repairs
    raw = auto_repair(raw)
    return raw


def _tokenize(s: str) -> list[str]:
    """Split into tokens: parens, brackets, braces, atoms."""
    return re.findall(r'[\(\)\[\]\{\}]|[^\s\(\)\[\]\{\}]+', s)


def _is_head_symbol(tok: str) -> bool:
    """True if token looks like a Neuralese HEAD (UPPER_CASE, 2+ chars, not a number)."""
    return bool(re.fullmatch(r'[A-Z][A-Z0-9_]+', tok))


def _find_matching(tokens: list[str], open_pos: int, open_ch: str, close_ch: str) -> int:
    """Return index of matching close bracket for open_ch at open_pos."""
    depth = 0
    for i in range(open_pos, len(tokens)):
        if tokens[i] == open_ch:
            depth += 1
        elif tokens[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return len(tokens) - 1


def _collect_top_level_exprs(tokens: list[str]) -> list[list[str]]:
    """Split a flat token list into top-level expression token groups (raw, no wrapping).
    Each group is either a bracketed sub-expression or a run of tokens starting with a HEAD.
    """
    exprs = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ('(', '[', '{'):
            close = {'(': ')', '[': ']', '{': '}'}[tok]
            end = _find_matching(tokens, i, tok, close)
            exprs.append(tokens[i:end + 1])
            i = end + 1
        elif _is_head_symbol(tok):
            # Collect bare HEAD + its args until next HEAD or bracket boundary
            group = [tok]
            i += 1
            while i < len(tokens) and tokens[i] not in (']', ')', '}') and not _is_head_symbol(tokens[i]):
                t = tokens[i]
                if t in ('(', '[', '{'):
                    close = {'(': ')', '[': ']', '{': '}'}[t]
                    end = _find_matching(tokens, i, t, close)
                    group.extend(tokens[i:end + 1])
                    i = end + 1
                else:
                    group.append(t)
                    i += 1
            exprs.append(group)  # raw, no wrapping — _repair_tokens will wrap
        else:
            exprs.append([tok])
            i += 1
    return exprs


def _tokens_to_str(tokens: list[str]) -> str:
    """Reassemble tokens into a clean S-expression string."""
    result = []
    for tok in tokens:
        if tok == ' ':
            # explicit space separator from multi-expr join — only add if needed
            if result and result[-1] not in ('(', '[', '{', ' '):
                result.append(' ')
        elif tok in (')', ']', '}'):
            # remove trailing space before close
            while result and result[-1] == ' ':
                result.pop()
            result.append(tok)
        elif tok in ('(', '[', '{'):
            # space before open-bracket unless at start or after another open-bracket
            if result and result[-1] not in ('(', '[', '{', ' '):
                result.append(' ')
            result.append(tok)
        else:
            # atom: space before unless at start or after open-bracket
            if result and result[-1] not in ('(', '[', '{', ' '):
                result.append(' ')
            result.append(tok)
    return ''.join(result)


def _repair_group(group: list[str]) -> list[str]:
    """
    Repair a single token group collected by _collect_top_level_exprs.
    - If already bracketed with ( or [, recurse inside without re-wrapping.
    - If a bare HEAD group, repair args then wrap in ().
    - Otherwise pass through (atoms, numbers).
    """
    if not group:
        return group

    if group[0] == '(':
        # Already parenthesized application (HEAD arg arg ...) — repair args only
        end = _find_matching(group, 0, '(', ')')
        interior = group[1:end]  # e.g. ['ARITH_ADD', '3', '4'] or ['HEAD', '(', 'sub', ')']
        if not interior:
            return ['(', ')']
        # Repair each arg token (recurse into nested brackets, pass atoms through)
        repaired_interior: list[str] = []
        i = 0
        while i < len(interior):
            tok = interior[i]
            if tok in ('(', '['):
                close = {'(': ')', '[': ']'}[tok]
                e = _find_matching(interior, i, tok, close)
                sub = _repair_group(interior[i:e + 1])
                if repaired_interior:
                    repaired_interior.append(' ')
                repaired_interior.extend(sub)
                i = e + 1
            else:
                if repaired_interior:
                    repaired_interior.append(' ')
                repaired_interior.append(tok)
                i += 1
        return ['('] + repaired_interior + [')']

    if group[0] == '[':
        # Sequence bracket — recurse interior, keep outer brackets
        end = _find_matching(group, 0, '[', ']')
        inner_repaired = _repair_sequence(group[1:end])
        return ['['] + inner_repaired + [']']

    if _is_head_symbol(group[0]):
        # Bare HEAD — repair args individually then wrap the whole thing in ()
        arg_tokens = group[1:]
        repaired_args: list[str] = []
        i = 0
        while i < len(arg_tokens):
            tok = arg_tokens[i]
            if tok in ('(', '['):
                close = {'(': ')', '[': ']'}[tok]
                end = _find_matching(arg_tokens, i, tok, close)
                sub = _repair_group(arg_tokens[i:end + 1])
                if repaired_args:
                    repaired_args.append(' ')
                repaired_args.extend(sub)
                i = end + 1
            elif _is_head_symbol(tok):
                # Nested bare HEAD in args — collect and repair recursively
                sub_group = [tok]
                i += 1
                while i < len(arg_tokens) and not _is_head_symbol(arg_tokens[i]) and arg_tokens[i] not in (']', ')'):
                    sub_group.append(arg_tokens[i])
                    i += 1
                repaired_sub = _repair_group(sub_group)
                if repaired_args:
                    repaired_args.append(' ')
                repaired_args.extend(repaired_sub)
            else:
                if repaired_args:
                    repaired_args.append(' ')
                repaired_args.append(tok)
                i += 1
        return ['(', group[0]] + ([' '] + repaired_args if repaired_args else []) + [')']

    return group  # atom or number — pass through


def _repair_sequence(tokens: list[str]) -> list[str]:
    """Collect top-level exprs from a token list, repair each, rejoin with spaces."""
    sub_exprs = _collect_top_level_exprs(tokens)
    result: list[str] = []
    for sub in sub_exprs:
        repaired = _repair_group(sub)
        if result:
            result.append(' ')
        result.extend(repaired)
    return result


def auto_repair(expr: str) -> str:
    """
    Fix the two most common structural errors from Qwen:
    1. Bare HEAD without parens: ARITH_ADD 3 4 → (ARITH_ADD 3 4)  [recursive]
    2. Multiple top-level expressions side-by-side: (A) (B) → [(A) (B)]
    """
    expr = expr.strip()
    tokens = _tokenize(expr)
    if not tokens:
        return expr

    top = _collect_top_level_exprs(tokens)

    if len(top) == 1:
        repaired = _repair_group(top[0])
        return _tokens_to_str(repaired)

    # Multiple top-level exprs — repair each then wrap in sequence [...]
    inner_tokens: list[str] = []
    for g in top:
        repaired = _repair_group(g)
        if inner_tokens:
            inner_tokens.append(' ')
        inner_tokens.extend(repaired)
    return '[' + _tokens_to_str(inner_tokens) + ']'


def assemble_chain(step_expressions: list[str]) -> str:
    """Join validated step expressions into a nested Sequence chain."""
    if not step_expressions:
        return ""
    if len(step_expressions) == 1:
        return step_expressions[0]
    result = step_expressions[-1]
    for expr in reversed(step_expressions[:-1]):
        result = f"[{expr} {result}]"
    return result


def count_tokens_approx(text: str) -> int:
    return len(text.split())


GRAMMAR_SUMMARY = (
    "Neuralese v1 Grammar:\n"
    "  Expr ::= Atom | (HEAD Expr*) | (λ VAR:TYPE . Expr) | {Expr | CONF} | [Expr Expr+]\n"
    "  Atom ::= SYMBOL | NUMBER | #VAR\n"
    "  HEAD ::= uppercase symbol (e.g. CAUSAL_ENABLES, LOGICAL_AND)\n"
    "  CONF ::= float 0.0–1.0 (only inside {expr | CONF} blocks)\n"
    "  VAR  ::= lowercase identifier\n"
    "  TYPE ::= Entity, Relation, Property, Process, Logical, Numeric, Modal\n\n"
    "Arithmetic operators (use THESE, not PLUS/MINUS/DIVIDE/EQUALS):\n"
    "  (ARITH_ADD a b)        (ARITH_SUBTRACT a b)    (ARITH_MULTIPLY a b)\n"
    "  (ARITH_DIVIDE a b)     (ARITH_EQUALS a b)      (ARITH_MODULO a b)\n"
    "  (ARITH_LESS_THAN a b)  (ARITH_GREATER_THAN a b)(ARITH_ASSIGN #var val)\n"
    "Numbers as arguments: use bare NUMBER literals — (ARITH_ADD 3 4) not (ARITH_ADD THREE FOUR)\n"
    "Percentages: 0.65 is a valid NUMBER atom — (ARITH_MULTIPLY #total 0.65)\n"
    "Variables: #var (lowercase after #) for intermediate values — NOT bare UPPERCASE like TOTAL\n"
    "  Special: NUMERIC_RESULT = final answer atom, RUNNING_TOTAL = accumulated sum atom\n\n"
    "Examples:\n"
    "  (CAUSAL_ENABLES STUDY KNOWLEDGE)\n"
    "  (ARITH_EQUALS (ARITH_ADD 3 4) #result)\n"
    "  [(ARITH_MULTIPLY 6 5) (ARITH_EQUALS NUMERIC_RESULT 30)]\n"
    "  {(IS_EQUIVALENT_TO NUMERIC_RESULT 42) | 1.0}"
)


def translate_example(
    client,
    question: str,
    cot: str,
    answer: str,
    primitives: list[dict],
    sigs,
) -> dict | None:
    original_tokens = count_tokens_approx(cot)

    steps = decompose_cot(client, question, cot)
    if not steps:
        return None

    validated_steps = []
    for step in steps:
        relevant = fuzzy_match_primitives(step, primitives)
        if not relevant:
            relevant = primitives[:10]

        expr = None
        parse_error: str = ""
        prior_candidate: str = ""
        for attempt in range(TRANSLATION_MAX_RETRIES):
            candidate = translate_step(
                client, step, relevant, GRAMMAR_SUMMARY,
                error_context=parse_error,
                prior_candidate=prior_candidate,
            )
            if candidate is None:
                break
            ok, err = validate(candidate, sigs)
            if ok:
                expr = candidate
                break
            # Pass the exact parser error so Ollama gets actionable feedback.
            # Generic retries tend to reproduce the same output — specific
            # error messages (ARITY_MISMATCH, UNKNOWN_SYMBOL, PARSE_ERROR)
            # reliably change the generation.
            parse_error = err or "INVALID_EXPRESSION"
            prior_candidate = candidate
            logger.debug(
                f"Step translation attempt {attempt + 1}/{TRANSLATION_MAX_RETRIES} "
                f"failed: {parse_error} | step: {step[:60]!r}"
            )

        if expr is None:
            return None

        validated_steps.append(expr)

    chain = assemble_chain(validated_steps)
    ok, err = validate(chain, sigs)
    if not ok:
        return None

    neuralese_tokens = count_tokens_approx(chain)
    compression_ratio = neuralese_tokens / max(original_tokens, 1)

    if compression_ratio > COMPRESSION_MAX_RATIO:
        return None

    # Discard trivially short CoTs that compress worse than English —
    # training on them teaches the model Neuralese is worse for simple arithmetic
    cot_words = len(cot.split())
    if cot_words < 25 and compression_ratio > 1.0:
        return None

    return {
        "question": question,
        "neuralese_chain": chain,
        "answer": answer,
        "original_cot": cot,
        "compression_ratio": round(compression_ratio, 4),
        "step_count": len(validated_steps),
    }


def load_dataset_examples(dataset_cfg: dict) -> list[tuple[str, str, str]]:
    """Returns list of (question, cot, answer) tuples."""
    name = dataset_cfg["name"]
    split = dataset_cfg["split"]
    n = dataset_cfg["n"]
    config = dataset_cfg.get("config")

    examples = []
    try:
        if config:
            ds = load_dataset(name, config, split=split)
        else:
            ds = load_dataset(name, split=split)

        for item in ds.shuffle(seed=42).select(range(min(n, len(ds)))):
            if name == "gsm8k":
                q = item["question"]
                cot_answer = item["answer"]
                parts = cot_answer.split("####")
                cot = parts[0].strip() if len(parts) > 1 else cot_answer
                ans = parts[1].strip() if len(parts) > 1 else ""
                examples.append((q, cot, ans))
            elif "ai2_arc" in name:
                q = item["question"]
                choices = item["choices"]
                ans_key = item["answerKey"]
                choice_labels = choices["label"]
                choice_texts = choices["text"]
                ans_idx = choice_labels.index(ans_key) if ans_key in choice_labels else 0
                ans_text = choice_texts[ans_idx]
                cot = q
                examples.append((q, cot, ans_text))
            elif "strategyqa" in name:
                q = item["question"]
                facts = item.get("facts", [])
                cot = " ".join(facts) if facts else q
                ans = "yes" if item.get("answer") else "no"
                examples.append((q, cot, ans))
            else:
                q = item.get("question", item.get("text", ""))
                cot = item.get("context", item.get("passage", q))
                ans = str(item.get("answer", item.get("label", "")))
                examples.append((q, cot, ans))

    except Exception as e:
        print(f"  Failed to load {name}: {e}")

    return examples


def main():
    require_phase(3)
    logger.info("Phase 4 — translate_dataset starting")

    primitives = load_primitives()
    sigs = get_signatures()
    client = get_client()

    logger.info(f"Loaded {len(primitives)} primitives, {len(sigs)} signatures.")

    # Count already-written examples for crash-resume — skip that many from the start.
    already_written = 0
    TRAIN_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TRAIN_CORPUS_PATH.exists():
        with open(TRAIN_CORPUS_PATH, encoding="utf-8") as _f:
            already_written = sum(1 for _ in _f)
        if already_written:
            logger.info(f"Resuming — {already_written} examples already on disk, skipping those.")

    total_written = already_written
    last_push_at = already_written  # track when we last pushed
    GIT_PUSH_EVERY = 100
    stats = {}

    # Open corpus file in append mode — each validated example written immediately.
    # Crash-safe: re-running skips already_written examples from the front of each dataset.
    with open(TRAIN_CORPUS_PATH, "a", encoding="utf-8") as corpus_f:
        skip_remaining = already_written  # global skip counter across all datasets

        for dataset_cfg in SOURCE_DATASETS:
            name = dataset_cfg["name"]
            print(f"\nProcessing {name}...")
            examples = load_dataset_examples(dataset_cfg)
            print(f"  Loaded {len(examples)} examples.")

            success, discarded = 0, 0

            try:
                for q, cot, ans in tqdm(examples, desc=f"  Translating {name}"):
                    if skip_remaining > 0:
                        skip_remaining -= 1
                        continue

                    result = translate_example(client, q, cot, ans, primitives, sigs)
                    if result is not None:
                        corpus_f.write(json.dumps(result) + "\n")
                        corpus_f.flush()
                        success += 1
                        total_written += 1
                        # Push to GitHub every GIT_PUSH_EVERY new examples
                        if total_written - last_push_at >= GIT_PUSH_EVERY:
                            last_push_at = total_written  # always advance to avoid retry storm
                            try:
                                repo = str(TRAIN_CORPUS_PATH.parent.parent)
                                subprocess.run(
                                    ["git", "-C", repo, "add", str(TRAIN_CORPUS_PATH)],
                                    check=True, capture_output=True
                                )
                                # Check if there's anything to commit
                                status = subprocess.run(
                                    ["git", "-C", repo, "status", "--porcelain"],
                                    check=True, capture_output=True
                                )
                                if status.stdout.strip():
                                    subprocess.run(
                                        ["git", "-C", repo, "commit", "-m",
                                         f"corpus checkpoint: {total_written} examples"],
                                        check=True, capture_output=True
                                    )
                                subprocess.run(
                                    ["git", "-C", repo, "push", "origin", "master"],
                                    check=True, capture_output=True
                                )
                                logger.info(f"Git push: {total_written} examples pushed to GitHub.")
                            except subprocess.CalledProcessError as e:
                                logger.warning(f"Git push failed at {total_written}: {e.stderr.decode().strip()[:200]}")
                    else:
                        discarded += 1
            except KeyboardInterrupt:
                logger.info(f"Interrupted during {name} — {total_written} examples saved so far.")
                break

            stats[name] = {
                "total": len(examples),
                "success": success,
                "discard_rate": round(1 - success / max(len(examples), 1), 3),
            }
            print(f"  {name}: {success}/{len(examples)} valid ({stats[name]['discard_rate']:.1%} discarded)")

    # Read back compression ratios from the written file for summary stats.
    all_ratios = []
    with open(TRAIN_CORPUS_PATH, encoding="utf-8") as _f:
        for line in _f:
            try:
                all_ratios.append(json.loads(line)["compression_ratio"])
            except (json.JSONDecodeError, KeyError):
                pass

    mean_compression = sum(all_ratios) / max(len(all_ratios), 1)
    stats["_summary"] = {
        "total_valid": total_written,
        "mean_compression_ratio": round(mean_compression, 4),
    }

    with open(CORPUS_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Corpus saved: {total_written} examples to {TRAIN_CORPUS_PATH}")
    logger.info(f"Mean compression ratio: {mean_compression:.3f}")
    if total_written < MIN_CORPUS_SIZE:
        logger.warning(
            f"Only {total_written} examples — target is ≥{MIN_CORPUS_SIZE}. "
            "Consider improving grammar coverage before Phase 5."
        )
    else:
        mark_complete(4, f"{total_written} examples, mean_ratio={mean_compression:.3f}")


if __name__ == "__main__":
    main()
