"""
Phase 3: Neuralese v1 parser + type checker.

Exposes:
  parse(expr_str)         → lark Tree or raises NeuraleseParseError
  typecheck(tree, sigs)   → True or raises NeuraleseTypeError
  validate(expr_str, sigs) → (ok: bool, error: str | None)
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from lark import Lark, Tree, Token
from lark.exceptions import UnexpectedInput

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRAMMAR_PATH, GRAMMAR_SIGNATURES_PATH


class NeuraleseParseError(Exception):
    pass


class NeuraleseTypeError(Exception):
    pass


@dataclass
class Signature:
    symbol: str
    arity: int
    domain: list[str]   # each entry is "Type1|Type2" union
    range: str


def load_grammar() -> Lark:
    grammar_text = GRAMMAR_PATH.read_text(encoding="utf-8")
    return Lark(grammar_text, parser="earley", ambiguity="resolve")


def load_signatures() -> dict[str, Signature]:
    if not GRAMMAR_SIGNATURES_PATH.exists():
        return {}
    with open(GRAMMAR_SIGNATURES_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    sigs = {}
    for entry in raw:
        sym = entry["symbol"]
        sigs[sym] = Signature(
            symbol=sym,
            arity=entry["arity"],
            domain=entry.get("domain", []),
            range=entry.get("range", "Unknown"),
        )
    return sigs


_grammar: Lark | None = None
_signatures: dict[str, Signature] | None = None


def get_grammar() -> Lark:
    global _grammar
    if _grammar is None:
        _grammar = load_grammar()
    return _grammar


def get_signatures() -> dict[str, Signature]:
    global _signatures
    if _signatures is None:
        _signatures = load_signatures()
    return _signatures


def parse(expr_str: str) -> Tree:
    try:
        return get_grammar().parse(expr_str.strip())
    except UnexpectedInput as e:
        raise NeuraleseParseError(f"PARSE_ERROR: {e}") from e


def _check_application(tree: Tree, sigs: dict[str, Signature]):
    """Type-check a single application node."""
    children = tree.children
    if not children:
        return

    head_token = None
    for child in children:
        if isinstance(child, Token) and child.type == "HEAD":
            head_token = str(child)
            break

    if head_token is None:
        return

    arg_exprs = [c for c in children if isinstance(c, Tree)]

    if not sigs:
        # No signatures loaded yet (Phase 3 hasn't run) — skip symbol checks,
        # only structural parse errors matter at this stage.
        return

    if head_token not in sigs:
        raise NeuraleseTypeError(f"UNKNOWN_SYMBOL: {head_token}")

    sig = sigs[head_token]
    if len(arg_exprs) != sig.arity:
        raise NeuraleseTypeError(
            f"ARITY_MISMATCH: {head_token} expects {sig.arity} args, got {len(arg_exprs)}"
        )


def typecheck(tree: Tree, sigs: dict[str, Signature] | None = None) -> bool:
    if sigs is None:
        sigs = get_signatures()

    for subtree in tree.iter_subtrees():
        if subtree.data == "application":
            _check_application(subtree, sigs)

    return True


def validate(expr_str: str, sigs: dict[str, Signature] | None = None) -> tuple[bool, str | None]:
    """Returns (True, None) on success or (False, error_message) on failure."""
    try:
        tree = parse(expr_str)
        typecheck(tree, sigs)
        return True, None
    except (NeuraleseParseError, NeuraleseTypeError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"UNEXPECTED_ERROR: {e}"


if __name__ == "__main__":
    import sys
    expr = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "(CAUSAL_ENABLES STUDY_PROCESS KNOWLEDGE_GAIN)"
    ok, err = validate(expr)
    if ok:
        print("VALID")
    else:
        print(f"INVALID: {err}")
