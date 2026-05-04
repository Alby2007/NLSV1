import sys
sys.path.insert(0, '.')
from phase4_corpus.translate_dataset import auto_repair

tests = [
    ("ARITH_MULTIPLY 40 2",
     "(ARITH_MULTIPLY 40 2)"),
    ("ARITH_MULTIPLY 40 2 ARITH_EQUALS #result 80",
     "[(ARITH_MULTIPLY 40 2) (ARITH_EQUALS #result 80)]"),
    ("(ARITH_MULTIPLY 6 2) (ARITH_EQUALS #total 12)",
     "[(ARITH_MULTIPLY 6 2) (ARITH_EQUALS #total 12)]"),
    ("(ARITH_ADD 3 4)",
     "(ARITH_ADD 3 4)"),
    ("[(ARITH_ADD 3 4) (ARITH_EQUALS #x 7)]",
     "[(ARITH_ADD 3 4) (ARITH_EQUALS #x 7)]"),
    ("(CAUSAL_ENABLES STUDY KNOWLEDGE) ARITH_ADD 3 4",
     "[(CAUSAL_ENABLES STUDY KNOWLEDGE) (ARITH_ADD 3 4)]"),
    # nested bare heads
    ("ARITH_ADD (ARITH_MULTIPLY 3 4) 5",
     "(ARITH_ADD (ARITH_MULTIPLY 3 4) 5)"),
    # bare HEAD inside sequence brackets
    ("[ARITH_SUBTRACT 50 10 ARITH_EQUALS 40]",
     "[(ARITH_SUBTRACT 50 10) (ARITH_EQUALS 40)]"),
    # nested sequences with bare HEADs
    ("[[ARITH_MULTIPLY 36 12 ARITH_EQUALS 432] [ARITH_DIVIDE 36 4 ARITH_EQUALS 9]]",
     "[[(ARITH_MULTIPLY 36 12) (ARITH_EQUALS 432)] [(ARITH_DIVIDE 36 4) (ARITH_EQUALS 9)]]"),
    # mixed: outer correct, inner bare
    ("[(ARITH_ADD 3 7) ARITH_DIVIDE 10 5]",
     "[(ARITH_ADD 3 7) (ARITH_DIVIDE 10 5)]"),
]

all_pass = True
for inp, expected in tests:
    got = auto_repair(inp)
    ok = got == expected
    if not ok:
        all_pass = False
    status = "OK  " if ok else "FAIL"
    print(f"  {status}  {inp[:55]}")
    if not ok:
        print(f"         expected: {expected}")
        print(f"         got:      {got}")

print()
print("ALL PASS" if all_pass else "FAILURES ABOVE — fix auto_repair()")
