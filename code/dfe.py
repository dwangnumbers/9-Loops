"""Double-final-entry relations: read them off the unique 3-loop answer, check the span equals the published
dim W_2 = 21, then impose them at four loops (the published amplitudes satisfy them through eight loops)."""
import numpy as np
from hexspace import HexSpace, P, nullspace_modp
from bootstrap import Bootstrap, matmul_modp as mm, rat
from bootstrap2 import Bootstrap2, BOUNDARY

def pair_vectors(H, c, n):
    """81 x d_{n-2} matrix: row (x,y) = coefficient vector of E^{x,y} (x then y = final entry)."""
    rows = []
    for x in range(9):
        for y in range(9):
            v = mm(mm(c[None, :], H.T[n][y]), H.T[n - 1][x])[0]
            rows.append(v)
    return np.array(rows)

def triple_vectors(H, c, n):
    rows = []
    for x in range(9):
        for y in range(9):
            for z in range(9):
                v = mm(mm(mm(c[None, :], H.T[n][z]), H.T[n - 1][y]), H.T[n - 2][x])[0]
                rows.append(v)
    return np.array(rows)

H = HexSpace('ext', verbose=False).build(10)
B2 = Bootstrap2(H)
c3, _, f3 = B2.run(3, conditions=('collinear', 'origin'), verbose=False)
V3 = pair_vectors(H, c3, 6)
rk3 = 81 - nullspace_modp(V3.T.copy() % P).shape[1]
print('L=3: span of double final entries =', rk3, '(published dim W_2 = 21)')
# four loops: particular solution + free direction
n = 8; T = H.T[n]; d = T.shape[1]
c0, _, free = B2.run(4, conditions=('collinear', 'origin'), verbose=False)
N = nullspace_modp(np.concatenate([T[0].T, T[1].T, T[2].T], axis=0) % P)
odd = np.nonzero(H.par[n] == 1)[0]; N = mm(N, nullspace_modp(N[odd] % P))
for g in ('cyc', 'flip'):
    G = B2.group_matrix(g, n); N = mm(N, nullspace_modp(mm((G.T - np.eye(d, dtype=np.int64)) % P, N)))
A = np.zeros((0, N.shape[1]), dtype=np.int64)
for cond in ('collinear', 'origin'):
    rows, _ = B2.sketch_rows(n, BOUNDARY[cond][0], N.shape[1] + 24); A = np.concatenate([A, mm(rows, N)])
cf = mm(N, nullspace_modp(A))[:, 0]
print('free directions at L=4 after collinear+origin:', nullspace_modp(A).shape[1])
V0, Vf = pair_vectors(H, c0, n), pair_vectors(H, cf, n)
print('L=4 spans: particular', 81 - nullspace_modp(V0.T.copy() % P).shape[1], ' free direction', 81 - nullspace_modp(Vf.T.copy() % P).shape[1])
if rk3 == 21:
    R = nullspace_modp(V3.T.copy() % P).T          # relations r (rows), r . V3 = 0
    a = mm(R, Vf).ravel(); b = mm(R, V0).ravel()
    nz = a != 0
    print('equations seeing the free direction:', int(nz.sum()))
    ts = set(int((-b[i]) * pow(int(a[i]), -1, P) % P) for i in np.nonzero(nz)[0])
    ok = all(b[i] == 0 for i in np.nonzero(~nz)[0])
    print('consistent:', ok and len(ts) == 1, ' t =', [rat(t) for t in ts][:3])
    if ok and len(ts) == 1:
        t = ts.pop(); c4 = (c0 + t * cf) % P
        np.save('E4.npy', c4)
        V4 = pair_vectors(H, c4, n)
        print('L=4 span of double final entries:', 81 - nullspace_modp(V4.T.copy() % P).shape[1])
        B = Bootstrap(H)
        print('E^(4): y-free terms', B.count_terms(c4, n, range(6)), '(published 11,208);  all terms', B.count_terms(c4, n, range(9)))
        # does the unique 4-loop answer satisfy the FF collinear, Euclidean MRK, SLL? (consistency)
        for cond in ('FF collinear', 'Euclidean MRK'):
            rows, Z = B2.sketch_rows(n, BOUNDARY[cond][0], 40)
            from bootstrap2 import quad_power, polarize
            from fractions import Fraction
            V, Q, sc = BOUNDARY[cond]
            Pf = quad_power([[Fraction(x) * sc for x in r] for r in Q], 4)
            lhs = mm(rows, c4[:, None])[:, 0]; rhs = np.array([polarize(Pf, z) for z in Z])
            print(f'   {cond} satisfied:', bool(np.all((lhs - rhs) % P == 0)))

def rank_mod(M):
    return M.shape[1] - nullspace_modp(M % P).shape[1]
import random
print('dim colspace(V3) =', rank_mod(V3.copy()))
for t in [0, 1, 7, random.randrange(P), random.randrange(P)]:
    M = np.concatenate([V3, (V0 + t * Vf) % P], axis=1)
    print('t =', t if t < 100 else 'random', ' dim(W(E3) + W(E4(t))) =', rank_mod(M.copy()))
