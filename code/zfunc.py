"""The weight-12 even function that vanishes on Delta=0: is it a valid 6-loop amplitude direction, and does the origin see it?"""
import numpy as np, time
from hexspace import HexSpace, P, nullspace_modp
from bootstrap import matmul_modp as mm
from bootstrap2 import Bootstrap2, BOUNDARY
from lift import annihilator
H = HexSpace('ext', verbose=False)
K = {1: np.zeros((3, 0), dtype=np.int64)}
for n in range(2, 13):
    H.step(n)
    Q = annihilator(K[n - 1], H.dim[n - 1])
    K[n] = nullspace_modp(np.concatenate([mm(Q, H.T[n][a].T.copy()) for a in range(6)], axis=0))
n = 12
even = np.nonzero(H.par[n] == 0)[0]; odd = np.nonzero(H.par[n] == 1)[0]
# even-only combinations of the vanishing subspace
Ke = mm(K[n], nullspace_modp(K[n][odd] % P))
print('even functions vanishing on Delta=0:', Ke.shape[1])
z = Ke[:, 0] % P
fe = [int(np.count_nonzero(mm(z[None, :], H.T[n][a]))) for a in range(3)]
print('MHV final entry (E^a=E^b=E^c=0):', fe == [0, 0, 0])
B2 = Bootstrap2(H)
for g in ('cyc', 'flip'):
    G = B2.group_matrix(g, n)
    gz = mm(G.T.copy(), z[:, None])[:, 0]
    sym = 'invariant' if np.array_equal(gz % P, z) else ('anti-invariant' if np.array_equal((gz + z) % P, np.zeros_like(z)) else 'mixes')
    print(f'{g}: {sym}')
for cond in ('collinear', 'origin'):
    rows, _ = B2.sketch_rows(n, BOUNDARY[cond][0], 8)
    print(f'{cond} limit sees it:', bool(np.any(mm(rows, z[:, None]) % P)))
