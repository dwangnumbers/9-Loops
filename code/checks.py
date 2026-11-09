"""Validate the miniature bootstrap against published results.
 1. E^(1) symbol = -1/2 (a(x)e + a(x)f + b(x)d + b(x)f + c(x)d + c(x)e)      [arXiv:1903.10890]
 2. E^(2) on Delta=0 = 1/2 a a a e + a e e e + dihedral                       [arXiv:2112.06243 eq. 22]
 3. #nonzero y-free words of E^(L) = #terms of the form factor F_3^(L): 12, 636, 11208, 263880
                                                                              [arXiv:2112.06243 Table II]
 4. the antipodal image (reverse words, a,b,c,d,e,f -> d,e,f,a,b,c, x2 per hatted a,b,c) of E^(2)
    equals 2 a a a e + 4 a e e e + dihedral, the published F_3^(2)           [arXiv:2112.06243 eq. 23]
"""
import sys, itertools, time, json
from fractions import Fraction
import numpy as np
from hexspace import HexSpace, P
from bootstrap import Bootstrap, rat, mm

NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'yu', 'yv', 'yw']
CYC6 = [1, 2, 0, 4, 5, 3]; FLIP6 = [1, 0, 2, 4, 3, 5]

def orbit(word):
    out = set()
    w = tuple(word)
    for _ in range(3):
        out.add(w); out.add(tuple(FLIP6[x] for x in w))
        w = tuple(CYC6[x] for x in w)
    return out

def symbol_dict(B, c, n, letters):
    """All words over `letters` with nonzero coefficient -> rational coefficient."""
    vecs = B.word_vectors(n, letters)
    words = list(vecs.keys())
    if not words:
        return {}
    M = np.array([vecs[w] for w in words])          # (#words, d_n)
    vals = mm(M, c[:, None])[:, 0]
    return {w: rat(int(v)) for w, v in zip(words, vals) if v}

def word_str(w):
    return ' '.join(NAMES[x] for x in w)

if __name__ == '__main__':
    Lmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    H = HexSpace('ext', verbose=False).build(2 * Lmax)
    B = Bootstrap(H)
    results = {}
    sols = {}
    for L in range(1, Lmax + 1):
        c, log = B.run(L, verbose=False)
        sols[L] = c
        nyfree = B.count_terms(c, 2 * L, range(6))
        nall = B.count_terms(c, 2 * L, range(9)) if L <= 4 else None
        results[L] = {'cascade': log, 'yfree_terms': nyfree, 'all_terms': nall}
        print(f'L={L}: cascade {[m for _, m in log]}, y-free terms {nyfree}, all terms {nall}  ({time.time()-t0:.1f}s)', flush=True)

    # 1. one loop
    S1 = symbol_dict(B, sols[1], 2, range(9))
    print('E^(1) symbol:', {word_str(w): str(v) for w, v in S1.items()})
    # 2. two loops on Delta = 0
    S2 = symbol_dict(B, sols[2], 4, range(6))
    pred = {}
    for w in orbit((0, 0, 0, 4)): pred[w] = Fraction(1, 2)
    for w in orbit((0, 4, 4, 4)): pred[w] = Fraction(1)
    print('E^(2)|Delta=0 matches 1/2 aaae + aeee + dihedral:', S2 == pred, f'({len(S2)} terms)')
    # 4. antipodal image of E^(2)|Delta=0 -> form factor
    amp2ff = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}       # hatted a->d, ..., hatted d->a
    FF2 = {}
    for w, v in S2.items():
        ffw = tuple(amp2ff[x] for x in reversed(w))
        # PRL normalisation: sqrt(a-hat) <-> d, i.e. factor 2 for each hatted a,b,c
        FF2[ffw] = v * 2 ** sum(1 for x in w if x < 3)
    predFF = {}
    for w in orbit((0, 0, 0, 4)): predFF[w] = 2
    for w in orbit((0, 4, 4, 4)): predFF[w] = 4
    print('antipode(E^(2)|Delta=0) == 2 aaae + 4 aeee + dihedral (form factor):', FF2 == predFF)
    results['E1'] = {word_str(w): str(v) for w, v in S1.items()}
    results['E2_delta0'] = {word_str(w): str(v) for w, v in S2.items()}
    with open('checks_out.json', 'w') as fh:
        json.dump({str(k): v for k, v in results.items()}, fh, indent=1, default=str)
