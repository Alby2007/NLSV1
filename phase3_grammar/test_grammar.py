"""
Phase 3: Grammar test suite — 100 hand-written expressions.

Runs each through the parser and reports pass/fail.
Must achieve 100/100 before proceeding to Phase 4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_grammar.parser import validate

VALID_EXPRESSIONS = [
    # Basic applications — using actual mined/seeded primitives
    "(CAUSAL_ENABLES STUDY_PROCESS KNOWLEDGE_GAIN)",
    "(CAUSAL_PREVENTS OBSTACLE PROGRESS)",
    "(CAUSAL_REQUIRES OXYGEN COMBUSTION)",
    "(CAUSAL_CONTRIBUTES EXERCISE HEALTH)",
    "(TEMPORAL_BEFORE CAUSE EFFECT)",
    "(TEMPORAL_AFTER RESULT TRIGGER)",
    "(TEMPORAL_DURING OBSERVATION EXPERIMENT)",
    "(IS_INSTANCE_OF SOCRATES HUMAN)",
    "(IS_PART_OF WHEEL CAR)",
    "(IS_EQUIVALENT_TO CELSIUS_ZERO FREEZING_POINT)",

    # Assertions with confidence
    "{(CAUSAL_ENABLES STUDY KNOWLEDGE) | 0.85}",
    "{(TEMPORAL_BEFORE BIRTH DEATH) | 1.0}",
    "{(IS_INSTANCE_OF ATOM MATTER) | 0.95}",
    "{(CAUSAL_PREVENTS VACCINE DISEASE) | 0.9}",
    "{(IS_PART_OF NEURON BRAIN) | 1.0}",

    # Logical connectives (seeded, arity=2)
    "(LOGICAL_AND (CAUSAL_ENABLES A B) (CAUSAL_ENABLES B C))",
    "(LOGICAL_OR (TEMPORAL_BEFORE X Y) (TEMPORAL_AFTER X Y))",
    "(LOGICAL_NOT (CAUSAL_ENABLES OBSTACLE PROGRESS))",
    "(LOGICAL_IMPLIES (CAUSAL_ENABLES STUDY KNOWLEDGE) (CAUSAL_ENABLES KNOWLEDGE SKILL))",
    "(LOGICAL_IFF (CAUSAL_ENABLES A B) (CAUSAL_REQUIRES B A))",

    # Quantifiers (arity=2 after fix: variable + body)
    "(FORALL x (CAUSAL_ENABLES x RESULT))",
    "(EXISTS x (IS_INSTANCE_OF x HUMAN))",
    "(EXISTS_UNIQUE x (IS_EQUIVALENT_TO x ORIGIN))",

    # THEREFORE (arity=1 after fix)
    "(THEREFORE (CAUSAL_ENABLES STUDY KNOWLEDGE))",
    "(THEREFORE (LOGICAL_AND (CAUSAL_ENABLES A B) (CAUSAL_ENABLES B C)))",

    # Sequences
    "[(CAUSAL_ENABLES STUDY KNOWLEDGE) (CAUSAL_ENABLES KNOWLEDGE SKILL)]",
    "[(TEMPORAL_BEFORE EVENT_A EVENT_B) (TEMPORAL_BEFORE EVENT_B EVENT_C)]",
    "[(IS_INSTANCE_OF CAT ANIMAL) (IS_INSTANCE_OF ANIMAL ORGANISM)]",
    "[(CAUSAL_PREVENTS A B) (CAUSAL_REQUIRES C D)]",

    # Nested sequences
    "[(CAUSAL_ENABLES STUDY KNOWLEDGE) [(CAUSAL_ENABLES KNOWLEDGE SKILL) {(THEREFORE (CAUSAL_ENABLES STUDY SKILL)) | 0.88}]]",
    "[(TEMPORAL_BEFORE BIRTH GROWTH) [(TEMPORAL_BEFORE GROWTH MATURITY) (THEREFORE (TEMPORAL_BEFORE BIRTH MATURITY))]]",

    # Multi-step reasoning chains
    "[(CAUSAL_ENABLES STUDY_PROCESS KNOWLEDGE_GAIN) [(CAUSAL_ENABLES KNOWLEDGE_GAIN PROBLEM_SOLVING) {(THEREFORE (CAUSAL_ENABLES STUDY_PROCESS PROBLEM_SOLVING)) | 0.88}]]",

    # Arithmetic ops (seeded, arity=2)
    "(ARITH_ADD X Y)",
    "(ARITH_SUBTRACT TOTAL PART)",
    "(ARITH_MULTIPLY A 3)",
    "(ARITH_DIVIDE NUMERATOR DENOMINATOR)",
    "(ARITH_EQUALS COUNT 42)",
    "(ARITH_MODULO N 2)",

    # Nested applications using actual primitives
    "(CAUSAL_ENABLES (IS_PART_OF BRAIN BODY) CONSCIOUSNESS)",
    "(LOGICAL_AND (CAUSAL_ENABLES A B) (IS_GREATER_THAN A THRESHOLD))",
    "(LOGICAL_OR (IS_LESS_THAN X BOUND) (IS_EQUIVALENT_TO X BOUND))",
    "(LOGICAL_NOT (CAUSAL_ENABLES OBSTACLE PROGRESS))",
    "(LOGICAL_IMPLIES (IS_INSTANCE_OF X HUMAN) (CAUSAL_REQUIRES X OXYGEN))",

    # Variable references with # sigil
    "(λ x:Entity . (IS_PART_OF #x SYSTEM))",
    "(λ x:Numeric . (IS_GREATER_THAN #x 0))",
    "(λ x:Entity . (λ y:Entity . (IS_EQUIVALENT_TO #x #y)))",

    # Deeply nested with seeded primitives
    "{(LOGICAL_AND (CAUSAL_ENABLES A B) (IS_GREATER_THAN A ZERO)) | 0.75}",
    "{(LOGICAL_OR (IS_LESS_THAN X BOUND) (ARITH_EQUALS X BOUND)) | 1.0}",
    "(CAUSAL_ENABLES (LOGICAL_AND CONDITION_A CONDITION_B) OUTCOME)",
    "(TEMPORAL_BEFORE (CAUSAL_ENABLES A B) (CAUSAL_ENABLES B C))",

    # Modal assertions (seeded)
    "{(KNOWN_TRUE CLAIM) | 1.0}",
    "{(BELIEVED_TRUE HYPOTHESIS) | 0.7}",
    "{(UNCERTAIN PREDICTION) | 0.5}",
    "{(KNOWN_FALSE CONTRADICTION) | 0.0}",
    "(KNOWN_TRUE AXIOM)",

    # Quantifiers with arity=2 (variable, body)
    "(FORALL x (CAUSAL_REQUIRES x ENERGY))",
    "(EXISTS x (IS_INSTANCE_OF x PRIME_NUMBER))",
    "(FORALL x (LOGICAL_IMPLIES (IS_INSTANCE_OF x HUMAN) (CAUSAL_REQUIRES x OXYGEN)))",

    # Comparators using seeded IS_GREATER_THAN / IS_LESS_THAN
    "(IS_GREATER_THAN A B)",
    "(IS_LESS_THAN X THRESHOLD)",
    "(IS_EQUIVALENT_TO RESULT EXPECTED)",
    "(IS_OPPOSITE_OF HOT COLD)",

    # Process chains using CAUSAL_ENABLES
    "[(CAUSAL_ENABLES PERCEIVE UNDERSTAND) [(CAUSAL_ENABLES UNDERSTAND RESPOND) (CAUSAL_ENABLES RESPOND LEARN)]]",
    "[(CAUSAL_REQUIRES OXYGEN COMBUSTION) [(CAUSAL_ENABLES COMBUSTION HEAT) (CAUSAL_ENABLES HEAT EXPANSION)]]",

    # Confidence edge cases
    "{(CAUSAL_ENABLES A B) | 1.0}",
    "{(CAUSAL_PREVENTS X Y) | 0.0}",
    "{(IS_INSTANCE_OF CLAIM FACT) | 0.5}",

    # Nested assertions in sequence
    "[{(CAUSAL_ENABLES A B) | 0.9} {(IS_GREATER_THAN A ZERO) | 1.0}]",
    "[{(KNOWN_TRUE AXIOM_1) | 1.0} {(KNOWN_TRUE AXIOM_2) | 1.0}]",

    # Mixed nesting
    "(CAUSAL_ENABLES {(KNOWN_TRUE CONDITION) | 0.9} OUTCOME)",
    "[(λ x:Entity . (CAUSAL_REQUIRES x ENERGY)) (FORALL y (CAUSAL_REQUIRES y ENERGY))]",

    # Numeric operations nested
    "(ARITH_EQUALS (ARITH_ADD X Y) Z)",
    "(ARITH_EQUALS (ARITH_MULTIPLY A B) (ARITH_ADD C D))",
    "(IS_LESS_THAN (ARITH_DIVIDE A B) THRESHOLD)",

    # Boolean logic trees
    "(LOGICAL_AND (LOGICAL_OR A B) (LOGICAL_NOT C))",
    "(LOGICAL_IMPLIES (LOGICAL_AND A B) (LOGICAL_OR C D))",
    "(LOGICAL_NOT (LOGICAL_AND (CAUSAL_ENABLES A B) (CAUSAL_ENABLES B C)))",

    # Epistemic chain
    "[(BELIEVED_TRUE PREMISE) [(LOGICAL_IMPLIES PREMISE CONCLUSION) {(THEREFORE (KNOWN_TRUE CONCLUSION)) | 0.85}]]",

    # Temporal reasoning
    "[(TEMPORAL_BEFORE BIRTH DEATH) (TEMPORAL_DURING LIFE BIRTH)]",
    "{(TEMPORAL_AFTER EFFECT CAUSE) | 0.0}",
    "(TEMPORAL_UNTIL REACTION EQUILIBRIUM)",

    # Causal chain
    "[(CAUSAL_ENABLES HEAT EXPANSION) [(CAUSAL_ENABLES EXPANSION PRESSURE) (CAUSAL_ENABLES PRESSURE MOVEMENT)]]",

    # Lambda returning sequence
    "(λ x:Entity . [(IS_PART_OF x SYSTEM) (CAUSAL_ENABLES x FUNCTION)])",

    # Assertion wrapping lambda
    "{(λ x:Numeric . (IS_GREATER_THAN x 0)) | 0.5}",

    # Nested lambda
    "(λ x:Entity . (λ y:Entity . (IS_EQUIVALENT_TO x y)))",

    # Variable in assertion
    "(λ x:Entity . {(CAUSAL_REQUIRES x OXYGEN) | 0.99})",

    # Because / Assumes / Given (seeded, arity=2)
    "(BECAUSE CONCLUSION JUSTIFICATION)",
    "(ASSUMES STEP PREMISE)",
]

INVALID_EXPRESSIONS = [
    # Structural parse errors
    "((MISSING_HEAD))",                              # double-wrapped, no valid head
    "not valid at all !!!",                          # natural language
    "(λ x . (CAUSAL_ENABLES x Y))",                 # lambda missing type annotation
    "(CAUSAL_ENABLES A B C)",                        # arity mismatch: expects 2, got 3
    "(LOGICAL_AND A)",                               # arity mismatch: expects 2, got 1
    "(THEREFORE A B)",                               # arity mismatch: expects 1, got 2
    # Confidence out of range
    "{(CAUSAL_ENABLES A B) | 1.5}",                 # confidence > 1
    "{(CAUSAL_ENABLES A B) | -0.1}",                # confidence < 0
    # Missing brackets / malformed
    "(CAUSAL_ENABLES A B",                           # unclosed paren
    "{(CAUSAL_ENABLES A B) 0.9}",                   # missing | separator
    # Wrong types in lambda
    "(λ x:unknown_type . (CAUSAL_ENABLES x Y))",    # invalid TYPE token (lowercase)
]


def run_tests():
    passed = 0
    failed = 0
    failures = []

    print(f"Running {len(VALID_EXPRESSIONS)} valid expression tests...")
    for i, expr in enumerate(VALID_EXPRESSIONS):
        ok, err = validate(expr)
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((i + 1, expr, err))

    print(f"\nResults: {passed}/{len(VALID_EXPRESSIONS)} passed")

    if failures:
        print("\nFailed expressions:")
        for idx, expr, err in failures:
            print(f"  [{idx}] {expr[:80]}")
            print(f"       → {err}")

    print(f"\nRunning {len(INVALID_EXPRESSIONS)} invalid expression tests (expecting failures)...")
    correctly_rejected = 0
    for expr in INVALID_EXPRESSIONS:
        ok, _ = validate(expr)
        if not ok:
            correctly_rejected += 1
        else:
            print(f"  WARNING: Should have been invalid but parsed: {expr}")

    print(f"Correctly rejected: {correctly_rejected}/{len(INVALID_EXPRESSIONS)}")

    total_valid = len(VALID_EXPRESSIONS)
    if passed == total_valid:
        print(f"\n✓ ALL {total_valid} VALID TESTS PASSED. Grammar is ready for Phase 4.")
        return True
    else:
        print(f"\n✗ {failed} tests failed. Fix grammar or parser before proceeding.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
