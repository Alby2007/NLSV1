"""
Hand-seeded primitives for underrepresented types.
These cover the connective tissue of reasoning chains that
activation mining underproduced due to pythia-1b's training data bias.
"""

SEED_PRIMITIVES = [
    # Relations — causal
    {"symbol": "CAUSAL_ENABLES",     "type": "Relation", "gloss": "X is sufficient to bring about Y"},
    {"symbol": "CAUSAL_PREVENTS",    "type": "Relation", "gloss": "X blocks or inhibits Y from occurring"},
    {"symbol": "CAUSAL_REQUIRES",    "type": "Relation", "gloss": "Y cannot occur without X"},
    {"symbol": "CAUSAL_CONTRIBUTES", "type": "Relation", "gloss": "X increases the likelihood of Y"},

    # Relations — temporal
    {"symbol": "TEMPORAL_BEFORE",    "type": "Relation", "gloss": "X occurs earlier in time than Y"},
    {"symbol": "TEMPORAL_AFTER",     "type": "Relation", "gloss": "X occurs later in time than Y"},
    {"symbol": "TEMPORAL_DURING",    "type": "Relation", "gloss": "X occurs within the time span of Y"},
    {"symbol": "TEMPORAL_UNTIL",     "type": "Relation", "gloss": "X continues up to the point of Y"},

    # Relations — logical/structural
    {"symbol": "IS_INSTANCE_OF",     "type": "Relation", "gloss": "X is a member of category Y"},
    {"symbol": "IS_PART_OF",         "type": "Relation", "gloss": "X is a constituent component of Y"},
    {"symbol": "IS_DEFINED_AS",      "type": "Relation", "gloss": "X has the meaning or specification Y"},
    {"symbol": "IS_EQUIVALENT_TO",   "type": "Relation", "gloss": "X and Y have the same value or meaning"},
    {"symbol": "IS_GREATER_THAN",    "type": "Relation", "gloss": "X exceeds Y in magnitude or degree"},
    {"symbol": "IS_LESS_THAN",       "type": "Relation", "gloss": "X is below Y in magnitude or degree"},
    {"symbol": "IS_OPPOSITE_OF",     "type": "Relation", "gloss": "X and Y are antithetical or contradictory"},

    # Logical — connectives
    {"symbol": "LOGICAL_AND",        "type": "Logical", "gloss": "Both X and Y hold simultaneously"},
    {"symbol": "LOGICAL_OR",         "type": "Logical", "gloss": "At least one of X or Y holds"},
    {"symbol": "LOGICAL_NOT",        "type": "Logical", "gloss": "X does not hold"},
    {"symbol": "LOGICAL_IMPLIES",    "type": "Logical", "gloss": "If X holds then Y necessarily holds"},
    {"symbol": "LOGICAL_IFF",        "type": "Logical", "gloss": "X holds if and only if Y holds"},
    {"symbol": "LOGICAL_XOR",        "type": "Logical", "gloss": "Exactly one of X or Y holds, not both"},

    # Logical — quantifiers
    {"symbol": "FORALL",             "type": "Logical", "gloss": "The following holds for every instance"},
    {"symbol": "EXISTS",             "type": "Logical", "gloss": "There is at least one instance for which this holds"},
    {"symbol": "EXISTS_UNIQUE",      "type": "Logical", "gloss": "There is exactly one instance for which this holds"},

    # Logical — reasoning steps
    {"symbol": "THEREFORE",         "type": "Logical", "gloss": "The preceding premises jointly entail this conclusion"},
    {"symbol": "BECAUSE",           "type": "Logical", "gloss": "The following is the justification for the prior claim"},
    {"symbol": "CONTRADICTS",       "type": "Logical", "gloss": "X and Y cannot both hold — they are mutually exclusive"},
    {"symbol": "ASSUMES",           "type": "Logical", "gloss": "The following reasoning step treats X as given"},
    {"symbol": "GIVEN",             "type": "Logical", "gloss": "X is established as a premise without derivation"},

    # Modal — epistemic
    {"symbol": "KNOWN_TRUE",        "type": "Modal", "gloss": "X is established with certainty"},
    {"symbol": "BELIEVED_TRUE",     "type": "Modal", "gloss": "X is held to be true but not proven"},
    {"symbol": "UNCERTAIN",         "type": "Modal", "gloss": "The truth value of X is not determined"},
    {"symbol": "KNOWN_FALSE",       "type": "Modal", "gloss": "X is established as not holding"},

    # Numeric — arithmetic operators
    {"symbol": "ARITH_ADD",         "type": "Numeric", "gloss": "The sum of X and Y"},
    {"symbol": "ARITH_SUBTRACT",    "type": "Numeric", "gloss": "The difference of X minus Y"},
    {"symbol": "ARITH_MULTIPLY",    "type": "Numeric", "gloss": "The product of X and Y"},
    {"symbol": "ARITH_DIVIDE",      "type": "Numeric", "gloss": "The quotient of X divided by Y"},
    {"symbol": "ARITH_EQUALS",      "type": "Numeric", "gloss": "X and Y are numerically identical"},
    {"symbol": "ARITH_MODULO",      "type": "Numeric", "gloss": "The remainder of X divided by Y"},
]

# Mined primitives too vague for stable grammar signatures — disabled before Phase 2
DISABLED_SYMBOLS = {
    "TOPICAL_CONCEPT",
    "BOTH_CONCEPT",
    "MATHEMATICAL_CONCEPT",
}
