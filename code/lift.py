"""
The lift off Delta = 0, in miniature.

On Delta = 0 all three y letters equal 1, so a symbol restricts to its y-free words.  Which hexagon
functions vanish there?  These are exactly the ambiguities that the 'lift' from the form factor
(which only knows Delta = 0) to the full amplitude must resolve.

Recursion:  c is in ker R_n  <=>  c T_n[alpha] lies in ker R_{n-1} for each of the six letters alpha.

Published (Cosmic9 conventions, sec. 5): the subspace of H_n vanishing on Delta=0 has dimensions
0,0,0,1,2,6,13,29,58,113,210,382,672,1164 (n = 0..13): the parity-odd part plus ONE parity-even
element at weight 12 and TWO at weight 13 (=> 2 x 111 = 222 parameters in the nine-loop lift).
"""
import sys, time
import numpy as np
from hexspace import HexSpace, P, nullspace_modp
from bootstrap import matmul_modp as mm

def mm_fast(A, B, p=P):
    """Exact (A @ B) mod p via float64 BLAS: split entries into 16-bit limbs; each limb product is exact
    as long as k * 2^32 < 2^53, i.e. inner dimension k < 2^21."""
    A = A % p; B = B % p
    assert A.shape[1] < 2**21
    A1, A0 = (A >> 16).astype(np.float64), (A & 0xFFFF).astype(np.float64)
    B1, B0 = (B >> 16).astype(np.float64), (B & 0xFFFF).astype(np.float64)
    hh = (A1 @ B1).astype(np.int64) % p
    mid = ((A1 @ B0).astype(np.int64) + (A0 @ B1).astype(np.int64)) % p
    ll = (A0 @ B0).astype(np.int64) % p
    return ((hh << 16) % p * 65536 % p + (mid << 16) % p + ll) % p

def annihilator(K, d):
    """Rows spanning the functionals that vanish on span(K) (K: d x k)."""
    if K.shape[1] == 0:
        return np.eye(d, dtype=np.int64)
    return nullspace_modp(K.T.copy() % P).T.copy()

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    H = HexSpace('ext', verbose=False)
    K = {1: np.zeros((3, 0), dtype=np.int64)}     # ln a, ln b, ln c do not vanish on Delta=0
    for n in range(2, N + 1):
        t0 = time.time()
        H.step(n)
        T = H.T[n]
        Q = annihilator(K[n - 1], H.dim[n - 1])  # (m x d_{n-1})
        A = np.concatenate([mm_fast(Q, T[a].T.copy()) for a in range(6)], axis=0)   # (6m x d_n)
        K[n] = nullspace_modp(A)
        print(f'weight {n:2d}: dim H_n = {H.dim[n]:5d}, vanishing on Delta=0: {K[n].shape[1]:5d} '
              f'(odd part {int((H.par[n]==1).sum()):4d}, even part {K[n].shape[1]-int((H.par[n]==1).sum())})'
              f'   [{time.time()-t0:.1f}s]', flush=True)
        for old in (n - 2,):                       # free memory we no longer need
            if old >= 2:
                H.T.pop(old, None); K.pop(old, None)
        del A, Q
