"""
Miniature hexagon bootstrap, version 2: boundary conditions imposed by *random rank-one sketches*.

A symbol-level limit condition says that a tensor X in (Q^k)^{(x) n} (the symbol of the limit) equals a
target tensor Y.  Instead of forming k^n entries we evaluate both sides on random rank-one test
tensors z_1 (x) ... (x) z_n with z_i in F_p^k:
      <X, z_1 (x) ... (x) z_n> = c . ( M_n ... M_1 [1] ),   M_w = sum_alpha (V_alpha . z_w) T_w[alpha]
i.e. every sampled equation costs n matrix-vector products.  A sampled system can only have a kernel
that is too large, never too small (it is a subset of the true equations); with p = 2^31-1 and
#samples > #unknowns + 20 it is exact with overwhelming probability.  This is the 'sketch' half of
the sketch-and-verify method used in the nine-loop computation.

Boundary data used (all at symbol level):
  collinear     S[E^(L)] -> S[(E1)^L/L!],  E1 -> -2 ln^2 T - 2 ln^2 S                    (T -> 0)
  origin        S[E^(L)] -> S[(E1)^L/L!],  E1 -> -1/2 (ln^2 u + ln^2 v + ln^2 w)         (u,v,w -> 0)
  FF collinear  via antipodal duality: E^(L) on Delta=0 is the antipode of the form factor E_c^(L),
                whose collinear limit is E_c^(L) -> (1/L!)(-ln^2 v - ln^2 d)^L            (v -> 0)
"""
import sys, time, itertools, math, random
from fractions import Fraction
import numpy as np
from hexspace import HexSpace, P, nullspace_modp
from bootstrap import matmul_modp as mm, rat, modp, CYC_PERM, CYC_SIGN, FLIP_PERM, FLIP_SIGN, COLL, ORIG

# form-factor collinear limit v -> 0 over the basis {v, u, 1-u}; amplitude letters on Delta=0 are
# a-hat = d^2, b-hat = e^2, c-hat = f^2, d-hat = a, e-hat = b, f-hat = c   (arXiv:2112.06243, eq. 21)
FF = {'a': (-1, 1, -1), 'b': (1, -1, -1), 'c': (-1, -1, 1), 'd': (0, -1, 1), 'e': (-1, 0, 0), 'f': (0, 1, -1)}
FFCOLL = np.array([[2*x for x in FF['d']], [2*x for x in FF['e']], [2*x for x in FF['f']],
                   FF['a'], FF['b'], FF['c'], (0, 0, 0), (0, 0, 0), (0, 0, 0)])

def quad_power(Q, L):
    """Return a function z -> (Q(z))^L / L! mod p, with Q a quadratic form given as a k x k Fraction matrix."""
    def f(z):
        q = 0
        for i in range(len(z)):
            for j in range(len(z)):
                if Q[i][j]:
                    q = (q + modp(Q[i][j]) * z[i] % P * z[j]) % P
        return pow(q, L, P) * pow(math.factorial(L), -1, P) % P
    return f

def polarize(Pfun, zs):
    """D_{z_1} ... D_{z_n} P for P homogeneous of degree n:  sum_S (-1)^{n-|S|} P(sum_{i in S} z_i).
    This equals <S[P], z_1 (x) ... (x) z_n>, the symbol of the polynomial-in-logs P contracted."""
    n, k = len(zs), len(zs[0])
    tot = 0
    for mask in range(1 << n):
        s = [0]*k
        bits = 0
        for i in range(n):
            if mask >> i & 1:
                bits += 1
                for j in range(k):
                    s[j] = (s[j] + zs[i][j]) % P
        val = Pfun(s)
        tot = (tot + (val if (n - bits) % 2 == 0 else -val)) % P
    return tot

BOUNDARY = {
    'collinear': (COLL, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], -2),   # -2(lT^2 + lS^2)
    'origin': (ORIG, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], Fraction(-1, 2)),
    'FF collinear': (FFCOLL, [[1, 0, 0], [0, 1, -1], [0, -1, 1]], -1),                  # -(lv^2 + (l(1-u)-lu)^2)
}


# Euclidean multi-Regge limit  u -> 1, v,w -> 0 :  u = 1-delta, v = delta/((1+w)(1+wb)), w = delta w wb/((1+w)(1+wb))
# basis {delta, w, wb, 1+w, 1+wb}; R_6 -> 0 there, so S[E^(L)] -> S[(E1)^L/L!], E1 -> -1/2 (ln^2 v + ln^2 w)
MRK = np.array([[-2, -1, -1, 2, 2], [0, -1, -1, 0, 0], [0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 1, 1],
                [-1, -1, -1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, -1, 1], [0, -1, 1, 1, -1]])
_A, _B = np.array([1, 0, 0, -1, -1]), np.array([1, 1, 1, -1, -1])
BOUNDARY['Euclidean MRK'] = (MRK, (np.outer(_A, _A) + np.outer(_B, _B)).tolist(), Fraction(-1, 2))

class Bootstrap2:
    def __init__(self, H, seed=7):
        self.H, self.p = H, H.p
        self.G = {'cyc': {0: np.array([[1]])}, 'flip': {0: np.array([[1]])}}
        self.rng = random.Random(seed)

    def group_matrix(self, name, n):
        G = self.G[name]
        if n in G:
            return G[n]
        perm, sign = (CYC_PERM, CYC_SIGN) if name == 'cyc' else (FLIP_PERM, FLIP_SIGN)
        Gm = self.group_matrix(name, n - 1)
        T, p = self.H.T[n], self.p
        U = np.zeros((9, T.shape[1], T.shape[2]), dtype=np.int64)
        for a in range(9):
            U[perm[a]] = (U[perm[a]] + sign[a] * mm(T[a], Gm)) % p
        free = self.H.free[n]
        G[n] = U[free % 9, :, free // 9].T % p
        return G[n]

    def sketch_rows(self, n, V, R):
        """R random rank-one functionals: returns (R x d_n) matrix of rows and the list of test vectors."""
        k = V.shape[1]
        Z = [[[self.rng.randrange(P) for _ in range(k)] for _ in range(n)] for _ in range(R)]
        Vp = V % P
        Y = np.ones((1, R), dtype=np.int64)
        for w in range(1, n + 1):
            T = self.H.T[w]
            s = np.array([[sum(int(Vp[a][j]) * Z[r][w - 1][j] for j in range(k)) % P for r in range(R)]
                          for a in range(9)], dtype=np.int64)                      # (9, R)
            Ynew = np.zeros((T.shape[1], R), dtype=np.int64)
            for a in range(9):
                if s[a].any():
                    Ynew = (Ynew + _mulcols(mm(T[a], Y), s[a])) % P
            Y = Ynew
        return Y.T.copy(), Z                                                        # (R, d_n)

    def run(self, L, conditions=('collinear', 'origin', 'FF collinear'), verbose=True, extra=24):
        n, p, T = 2 * L, self.p, self.H.T[2 * L]
        d = T.shape[1]
        log = [('ansatz in hexagon space', d)]
        N = nullspace_modp(np.concatenate([T[0].T, T[1].T, T[2].T], axis=0) % p)
        log.append(('MHV final entry', N.shape[1]))
        odd = np.nonzero(self.H.par[n] == 1)[0]
        if len(odd):
            N = mm(N, nullspace_modp(N[odd] % p))
        log.append(('parity', N.shape[1]))
        for g in ('cyc', 'flip'):
            G = self.group_matrix(g, n)
            N = mm(N, nullspace_modp(mm((G.T - np.eye(d, dtype=np.int64)) % p, N)))
        log.append(('dihedral symmetry', N.shape[1]))
        m = N.shape[1]
        A_all, b_all = np.zeros((0, m), dtype=np.int64), np.zeros(0, dtype=np.int64)
        for cond in conditions:
            V, Q, scale = BOUNDARY[cond]
            Qs = [[Fraction(x) * scale for x in row] for row in Q]
            Pfun = quad_power(Qs, L)
            rows, Z = self.sketch_rows(n, V, m + extra)
            A = mm(rows, N)
            b = np.array([polarize(Pfun, z) for z in Z], dtype=np.int64)
            A_all = np.concatenate([A_all, A]); b_all = np.concatenate([b_all, b])
            K = nullspace_modp(np.concatenate([A_all, (-b_all % p)[:, None]], axis=1))
            if not (K[-1] % p).any():
                raise RuntimeError(f'L={L}: boundary data inconsistent at {cond}')
            log.append((cond, K.shape[1] - 1))
        j = int(np.nonzero(K[-1] % p)[0][0])
        x = K[:-1, j] * pow(int(K[-1, j]), -1, p) % p
        c = mm(N, x[:, None])[:, 0]
        if verbose:
            for name, mm_ in log:
                print(f'   {name:28s} {mm_}', flush=True)
        return c, log, K.shape[1] - 1

def _mulcols(A, s):
    """A * s (column scaling) mod p, exact: split s into 16-bit halves."""
    s = s % P
    lo, hi = s & 0xFFFF, s >> 16
    return ((A * lo[None, :]) % P + ((A * hi[None, :]) % P << 16) % P) % P

if __name__ == '__main__':
    Lmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    H = HexSpace('ext', verbose=False).build(2 * Lmax)
    B = Bootstrap2(H)
    for L in range(1, Lmax + 1):
        t0 = time.time()
        print(f'L = {L}:')
        B.run(L, conditions=('collinear', 'origin', 'FF collinear', 'Euclidean MRK'))
        print(f'   ({time.time()-t0:.1f}s)')
