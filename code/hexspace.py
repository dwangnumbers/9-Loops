"""
Build the hexagon symbol spaces H_n recursively, exactly over F_p (p = 2^31 - 1).

  H_n = { sum_x F^x (x) x  :  F^x in H_{n-1},  41-ish pair relations hold on the last two slots }

A basis of H_n is stored as nested coproduct tensors T_n[alpha] (d_n x d_{n-1}):
      F_i^(n) = sum_{j, alpha} T_n[alpha][i, j]  F_j^(n-1) (x) x_alpha
which is exactly the representation used for the nine-loop computation (arXiv:1906.07116 sec. 4,
and the 'nested coproduct basis' of the Cosmic9 data page).

modes:  'first'     first-entry condition + integrability only
        'steinmann' + Steinmann relations on the first two slots only
        'ext'       + extended Steinmann relations in every pair of adjacent slots
"""
import ctypes, os, sys, time, pickle
import numpy as np
import sympy as sp
from alphabet import integrability_relations, steinmann_relations, ODD

P = 2147483647
HERE = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(HERE, 'librref.so'))
_lib.rref_modp.restype = ctypes.c_int
_lib.rref_modp.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int,
                           ctypes.c_int64, ctypes.POINTER(ctypes.c_int)]

_lib.rref_modp32.restype = ctypes.c_int
_lib.rref_modp32.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int,
                             ctypes.c_int64, ctypes.POINTER(ctypes.c_int)]

def nullspace_modp32(M32, p=P):
    """Same as nullspace_modp but M32 is a uint32 array that is destroyed (lower memory)."""
    rows, cols = M32.shape
    if rows == 0:
        return np.eye(cols, dtype=np.int64)
    piv = (ctypes.c_int * min(rows, cols))()
    r = _lib.rref_modp32(M32.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), rows, cols, p, piv)
    pivots = np.array([piv[i] for i in range(r)], dtype=np.int64)
    free = np.setdiff1d(np.arange(cols), pivots)
    K = np.zeros((cols, len(free)), dtype=np.int64)
    K[free, np.arange(len(free))] = 1
    K[pivots[:, None], np.arange(len(free))[None, :]] = (p - M32[:r][:, free].astype(np.int64)) % p
    nullspace_modp32.last_free = free
    return K

def to_modp(q, p=P):
    q = sp.Rational(q)
    return (int(q.p) % p) * pow(int(q.q) % p, -1, p) % p

def nullspace_modp(M, p=P):
    """Kernel of M (rows x cols) over F_p. Returns K (cols x k) in 'pivot form':
    each kernel vector has a 1 in its own free column and 0 in the other free columns."""
    M = np.ascontiguousarray(M % p, dtype=np.int64)
    rows, cols = M.shape
    if rows == 0:
        return np.eye(cols, dtype=np.int64)
    piv = (ctypes.c_int * min(rows, cols))()
    r = _lib.rref_modp(M.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), rows, cols, p, piv)
    pivots = [piv[i] for i in range(r)]
    free = sorted(set(range(cols)) - set(pivots))
    K = np.zeros((cols, len(free)), dtype=np.int64)
    R = M[:r]
    for t, fcol in enumerate(free):
        K[fcol, t] = 1
        K[pivots, t] = (-R[:, fcol]) % p
    return K

def relation_tensors(mode_steinmann):
    rels = integrability_relations()
    if mode_steinmann:
        rels = rels + steinmann_relations()
    D = np.array([[to_modp(x) for x in r] for r in rels], dtype=np.int64).reshape(len(rels), 9, 9)
    return D

class HexSpace:
    def __init__(self, mode='ext', p=P, verbose=True):
        self.mode, self.p, self.verbose = mode, p, verbose
        self.D_int = relation_tensors(False)
        self.D_ste = relation_tensors(True)
        # weight 1: ln a, ln b, ln c   (first-entry condition)
        T1 = np.zeros((9, 3, 1), dtype=np.int64)
        for i in range(3):
            T1[i, i, 0] = 1
        self.T = {1: T1}
        self.par = {0: np.array([0]), 1: np.array([0, 0, 0])}
        self.dim = {0: 1, 1: 3}
        self.free = {1: np.array([0, 1, 2])}

    def relations_for(self, n):
        if self.mode == 'ext' or (self.mode == 'steinmann' and n == 2):
            return self.D_ste
        return self.D_int

    def step(self, n):
        p = self.p
        Tm = self.T[n - 1]                      # (9, d_{n-1}, d_{n-2})
        d1, d2 = Tm.shape[1], Tm.shape[2]
        D = self.relations_for(n)
        parprev = self.par[n - 1]
        colpar = (parprev[:, None] ^ np.array(ODD)[None, :]).reshape(-1)   # parity of (j,beta)
        Tn_parts, par_parts, free_parts = [], [], []
        for target in (0, 1):
            cols = np.nonzero(colpar == target)[0]
            if len(cols) == 0:
                continue
            # M[(m,k), (j,beta)] = sum_alpha D[m,alpha,beta] T_{n-1}[alpha][j,k], built block by block
            blocks = []
            for m in range(D.shape[0]):
                B = np.zeros((d2, d1, 9), dtype=np.int64)
                for a in range(9):
                    for b in range(9):
                        if D[m, a, b]:
                            B[:, :, b] = (B[:, :, b] + D[m, a, b] * Tm[a].T) % p
                B = B.reshape(d2, d1 * 9)[:, cols]
                B = B[np.any(B != 0, axis=1)]
                blocks.append(B.astype(np.uint32))
                del B
            Ms = np.ascontiguousarray(np.concatenate(blocks, axis=0)); del blocks
            shape = Ms.shape
            t0 = time.time()
            K = nullspace_modp32(Ms, p)         # (len(cols), k)
            del Ms
            if self.verbose:
                print(f'  weight {n} parity {"odd" if target else "even"}: system {shape[0]} x {shape[1]}'
                      f' -> kernel {K.shape[1]}  ({time.time()-t0:.1f}s)', flush=True)
            free_parts.append(cols[nullspace_modp32.last_free])
            full = np.zeros((d1 * 9, K.shape[1]), dtype=np.int64)
            full[cols] = K
            Tn_parts.append(full)
            par_parts.append(np.full(K.shape[1], target))
        N = np.concatenate(Tn_parts, axis=1)    # (d1*9, d_n)
        dn = N.shape[1]
        Tn = N.reshape(d1, 9, dn).transpose(1, 2, 0).copy()   # (9, d_n, d_{n-1})
        self.T[n] = Tn
        self.par[n] = np.concatenate(par_parts)
        self.free[n] = np.concatenate(free_parts)      # pivot column (j*9+beta) of each basis element
        self.dim[n] = dn
        return dn

    def build(self, nmax):
        for n in range(2, nmax + 1):
            self.step(n)
            if self.verbose:
                ev = int((self.par[n] == 0).sum())
                print(f'dim H_{n} = {self.dim[n]}  (even {ev}, odd {self.dim[n]-ev})', flush=True)
        return self

    # ---- utilities -------------------------------------------------------------------------
    def word_coefficient(self, n, i, word):
        """Coefficient of the word x_1...x_n (letter indices, first entry first) in F_i^(n)."""
        v = np.zeros(self.dim[n], dtype=np.int64); v[i] = 1
        for k in range(n, 0, -1):
            v = (v @ self.T[k][word[k - 1]]) % self.p
        return int(v[0])

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'ext'
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    H = HexSpace(mode).build(nmax)
    print(mode, [H.dim[n] for n in range(nmax + 1)])
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'wb') as fh:
            pickle.dump({'T': H.T, 'par': H.par, 'dim': H.dim}, fh)
