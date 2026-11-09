"""
Quick demo (runs in a few seconds):

    python3 demo.py

 1. pair relations of the hexagon alphabet (26 integrability, 55 integrable pairs)
 2. dimensions of the extended-Steinmann hexagon spaces H_n (Table 1 of arXiv:1906.07116)
 3. bootstrap of E^(1), E^(2), E^(3) from scratch, with the parameter cascade
 4. checks against arXiv:2112.06243: E^(2) on Delta=0, its antipode = the 2-loop form factor,
    and the term counts 12 and 636 of the form factor
"""
import time
from fractions import Fraction
import sympy as sp
from alphabet import integrability_relations, steinmann_relations
from hexspace import HexSpace
from bootstrap import Bootstrap
from bootstrap2 import Bootstrap2
from checks import symbol_dict, orbit, word_str

def banner(t):
    print('\n' + t + '\n' + '-' * len(t))

t0 = time.time()
banner('1. Not every tensor is a symbol')
I = integrability_relations()
print(f'integrability relations among the 81 letter pairs: {len(I)}  ->  {81-len(I)} integrable pairs')
r = sp.Matrix(I + steinmann_relations()).rank()
print(f'+ extended Steinmann: rank {r}  ->  {81-r} allowed pairs')

banner('2. The hexagon function spaces H_n, built as iterated kernels over F_p, p = 2^31 - 1')
H = HexSpace('ext', verbose=False)
for n in range(2, 10):
    H.step(n)
print('dim H_n, n = 0..9 :', [H.dim[n] for n in range(10)])
print('published         :', [1, 3, 6, 13, 26, 51, 98, 184, 340, 613])

banner('3. Bootstrap: cross out what physics forbids')
B2 = Bootstrap2(H)
stages = None
sols = {}
for L in (1, 2, 3):
    c, log, free = B2.run(L, conditions=('collinear', 'origin'), verbose=False)
    sols[L] = c
    if stages is None:
        stages = [name for name, _ in log]
        print(f'{"":28s}' + ''.join(f'{"L="+str(l):>7s}' for l in (1, 2, 3)))
        rows = {s: [] for s in stages}
    for name, m in log:
        rows[name].append(m)
for s in stages:
    print(f'{s:28s}' + ''.join(f'{m:7d}' for m in rows[s]))
print('-> unique answer at every loop order shown')

banner('4. Compare with the literature')
B = Bootstrap(H)
S1 = symbol_dict(B, sols[1], 2, range(9))
print('S[E^(1)] =', ' '.join(f'{v:+} {word_str(w).replace(" ", "x")}' for w, v in sorted(S1.items())))
S2 = symbol_dict(B, sols[2], 4, range(6))
pred = {w: Fraction(1, 2) for w in orbit((0, 0, 0, 4))}
pred.update({w: Fraction(1) for w in orbit((0, 4, 4, 4))})
print(f'E^(2) on Delta=0 has {len(S2)} terms;  equals 1/2 aaae + aeee + dihedral (PRL eq. 22):', S2 == pred)
amp2ff = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
FF2 = {tuple(amp2ff[x] for x in reversed(w)): v * 2 ** sum(1 for x in w if x < 3) for w, v in S2.items()}
predFF = {w: 2 for w in orbit((0, 0, 0, 4))}
predFF.update({w: 4 for w in orbit((0, 4, 4, 4))})
print('antipode -> form factor 2 aaae + 4 aeee + dihedral (PRL eq. 23):', FF2 == predFF)
n3 = B.count_terms(sols[3], 6, range(6))
print(f'terms of the 3-loop form factor via the duality: {n3}   (published: 636)')
print(f'\ntotal time {time.time()-t0:.1f}s')
