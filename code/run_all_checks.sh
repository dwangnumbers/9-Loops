#!/usr/bin/env bash
# Reproduce every number this repository claims to compute (about 5 minutes; cascade6/lift up to weight 13 are separate).
# Expected values are given in the comments.
set -e
cd "$(dirname "$0")"
[ -f librref.so ] || gcc -O3 -shared -fPIC -o librref.so rref.c

echo "== demo (26 relations; dims 1..613; unique E1,E2,E3; PRL eqs. 22/23 True; 636 terms)"
python3 demo.py

echo "== checks 4  (y-free terms 6, 12, 636; full symbols 6, 84, 5034; L=4 leaves 1 free direction)"
python3 checks.py 4 | head -4

echo "== multi-Regge super-leading-log identity  (True at L=2,3; 0 new equations at L=4)"
python3 mrk_sll.py | grep -E "SLL"

echo "== four-loop term count  (generic value of the free direction gives 11208; 25 special values)"
python3 fix_by_count.py | tail -2

echo "== adjacency  (pair space 40; 12 forbidden pairs)"
python3 fig_pairs.py

echo "== fill-in  (kernel 282; peak nnz 73,147 natural / 197,449 random / 108,281 sparsest-first)"
python3 fig_fillin.py

echo "== the weight-12 even function vanishing on Delta=0  (final entry True; D3 invariant; collinear False; origin True)"
python3 zfunc.py

# Longer runs:
#   python3 hexspace.py ext 13   -> dim H_13 = 5431 (4269 + 1162), ~17 min
#   python3 lift.py 13           -> even functions vanishing on Delta=0: 0 through weight 11, 1 at 12, 2 at 13, ~25 min
#   python3 cascade6.py          -> L=6 cascade 3224, 1292, 885, 197, 14, 11, 11, 11, ~5 min
