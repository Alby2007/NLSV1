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
from config import GRAMMAR_PATH, GRAMMAR_SIGNATURES_PATH, ONTOLOGY_PATH


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


# ── Strict domain/range type checking ────────────────────────────────────────
# Loose typecheck only enforces arity + symbol existence. Strict mode also
# verifies that each argument's inferred type matches the signature's domain
# (with subtype rules: a domain of "Entity" accepts "ConcreteEntity", not vice
# versa). Un-inferable nodes return None and pass leniently.

_ontology: dict[str, list[str]] | None = None


def get_ontology() -> dict[str, list[str]]:
    """symbol -> type_path, e.g. CONCRETE_THING -> ['Entity', 'ConcreteEntity']."""
    global _ontology
    if _ontology is None:
        _ontology = {}
        if ONTOLOGY_PATH.exists():
            for p in json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8")):
                _ontology[p["symbol"]] = p.get("type_path") or [p.get("type", "Unknown")]
    return _ontology


def _infer_path(node, sigs: dict[str, Signature], ontology: dict[str, list[str]]) -> list[str] | None:
    """Infer the type path of an expr subtree. None = can't determine."""
    if isinstance(node, Token):
        return None
    if node.data == "atom":
        for tok in node.children:
            if isinstance(tok, Token):
                if tok.type == "NUMBER":
                    return ["Numeric"]
                if tok.type == "SYMBOL":
                    return ontology.get(str(tok))
        return None
    if node.data == "application":
        head = next((str(c) for c in node.children if isinstance(c, Token) and c.type == "HEAD"), None)
        if head and head in sigs:
            return [sigs[head].range]
        return None
    if node.data == "assertion":
        inner = next((c for c in node.children if isinstance(c, Tree)), None)
        return _infer_path(inner, sigs, ontology) if inner is not None else None
    if node.data == "sequence":
        return ["Sequence"]
    return None


def _check_application_strict(tree: Tree, sigs: dict[str, Signature], ontology: dict[str, list[str]]):
    """Arity + domain type check for a single application node."""
    _check_application(tree, sigs)  # arity + UNKNOWN_SYMBOL first

    head_token = next((str(c) for c in tree.children if isinstance(c, Token) and c.type == "HEAD"), None)
    if not head_token or head_token not in sigs:
        return
    sig = sigs[head_token]
    arg_exprs = [c for c in tree.children if isinstance(c, Tree)]

    for i, (arg, domain_entry) in enumerate(zip(arg_exprs, sig.domain)):
        arg_path = _infer_path(arg, sigs, ontology)
        if arg_path is None:
            continue  # lenient on un-inferable
        allowed = domain_entry.split("|") if isinstance(domain_entry, str) else [domain_entry]
        if not any(m in arg_path for m in allowed):
            raise NeuraleseTypeError(
                f"TYPE_ERROR: {head_token} arg{i+1} expects {domain_entry}, got {'/'.join(arg_path)}"
            )


def typecheck_strict(tree: Tree, sigs: dict[str, Signature] | None = None) -> bool:
    if sigs is None:
        sigs = get_signatures()
    ontology = get_ontology()

    for subtree in tree.iter_subtrees():
        if subtree.data == "application":
            _check_application_strict(subtree, sigs, ontology)

    return True


def validate_strict(expr_str: str, sigs: dict[str, Signature] | None = None) -> tuple[bool, str | None]:
    """Strict validation: arity + UNKNOWN_SYMBOL + domain/range types."""
    try:
        tree = parse(expr_str)
        typecheck_strict(tree, sigs)
        return True, None
    except (NeuraleseParseError, NeuraleseTypeError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"UNEXPECTED_ERROR: {e}"


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
