"""
Fix the last four-loop parameter with the multi-Regge super-leading-logarithm (SLL) identity:
in 2->4 multi-Regge kinematics the u-discontinuity of the remainder function satisfies
        Disc_u R^(L)  ~  sum_{r <= L-1} ln^r(1-u) (...)
so every word of S[Disc_u R^(L)]|_MRK containing >= L copies of the letter delta = 1-u must cancel.
(arXiv:1903.10890 sec. 4; the same check is applied to the 6-8 loop lifts in the Cosmic9 notes.)

R^(L) = (ln E)^(L) at symbol level (Gamma_cusp/4 -> g^2, rho invisible):
  R^(2) = E2 - E1^2/2
  R^(3) = E3 - E1 E2 + E1^3/3
  R^(4) = E4 - E1 E3 - E2^2/2 + E1^2 E2 - E1^4/4
Disc_u acts on the first entry: ln a-hat -> +1, ln b-hat -> -1, ln c-hat -> -1 (coefficient of ln u);
it is a derivation of the shuffle product.  The MRK limit replaces letters by their leading monomials.
"""
import sys, time, itertools, json
from fractions import Fraction
import numpy as np
from hexspace import HexSpace, P, nullspace_modp
from bootstrap import matmul_modp as mm, rat, modp
from bootstrap2 import Bootstrap2, MRK, BOUNDARY

KAPPA = {0: 1, 1: -1, 2: -1}          # coefficient of ln u in ln a, ln b, ln c
K5 = MRK.shape[1]                    # limit alphabet {delta, w, wb, 1+w, 1+wb}

def lim_tensors(H, c, n):
    """Return (lim F, lim Disc_u F) as dicts word->residue over the 5-letter MRK alphabet (dense top-down)."""
    p = P
    S = (c % p)[None, :]
    ln = 0
    for w in range(n, 1, -1):
        T = H.T[w]
        new = np.zeros((K5 ** (ln + 1), T.shape[2]), dtype=np.int64)
        blk = K5 ** ln
        for a in range(9):
            if not MRK[a].any():
                continue
            V = mm(S, T[a])
            for j in range(K5):
                e = int(MRK[a][j])
                if e:
                    new[j * blk:(j + 1) * blk] = (new[j * blk:(j + 1) * blk] + (e % p) * V) % p
        S, ln = new, ln + 1
    T1 = H.T[1]
    lim = np.zeros(K5 ** (ln + 1), dtype=np.int64); disc = np.zeros(K5 ** ln, dtype=np.int64)
    blk = K5 ** ln
    for a in range(3):
        coef = mm(S, T1[a])[:, 0]
        for j in range(K5):
            e = int(MRK[a][j])
            if e:
                lim[j * blk:(j + 1) * blk] = (lim[j * blk:(j + 1) * blk] + (e % p) * coef) % p
        disc = (disc + (KAPPA[a] % p) * coef) % p
    def todict(arr, length):
        idx = np.nonzero(arr)[0]
        words = np.array(np.unravel_index(idx, (K5,) * length)).T if length else np.zeros((len(idx), 0), int)
        return {tuple(int(x) for x in wd): int(arr[i]) for wd, i in zip(words, idx)}
    return todict(lim, ln + 1), todict(disc, ln)

def shuffle(A, B):
    """Shuffle product of two symbols given as dicts word->coeff (mod p)."""
    out = {}
    for u, cu in A.items():
        for v, cv in B.items():
            m, n = len(u), len(v)
            cc = cu * cv % P
            for pos in itertools.combinations(range(m + n), m):
                w = [None] * (m + n); iu = iv = 0; ps = set(pos)
                for k in range(m + n):
                    if k in ps:
                        w[k] = u[iu]; iu += 1
                    else:
                        w[k] = v[iv]; iv += 1
                w = tuple(w)
                out[w] = (out.get(w, 0) + cc) % P
    return {k: v for k, v in out.items() if v}

def add(*terms):
    out = {}
    for coef, D in terms:
        c = modp(coef)
        for k, v in D.items():
            out[k] = (out.get(k, 0) + c * v) % P
    return {k: v for k, v in out.items() if v}

def disc_product(factors):
    """lim Disc(f1 f2 ... fk) = sum_i  lim f1 sh ... sh lim Disc f_i sh ... sh lim fk"""
    total = {}
    for i in range(len(factors)):
        acc = {(): 1}
        for j, (lim, disc) in enumerate(factors):
            acc = shuffle(acc, disc if j == i else lim)
        total = add((1, total), (1, acc))
    return total

def high_delta(D, L):
    return {k: v for k, v in D.items() if k.count(0) >= L}

if __name__ == '__main__':
    L = 4
    H = HexSpace('ext', verbose=False).build(2 * L)
    B = Bootstrap2(H)
    sols = {}
    for l in range(1, L):
        sols[l], _, free = B.run(l, verbose=False)
        assert free == 0
    # four loops: particular solution + free direction
    c0, log, free = B.run(L, conditions=('collinear', 'origin', 'FF collinear'), verbose=True)
    # recover the free direction: difference of two particular solutions with different sketch seeds
    B.rng.seed(12345)
    # obtain a second, independent particular solution by adding a random kernel element
    n = 2 * L
    # rebuild the homogeneous solution space explicitly
    T = H.T[n]; d = T.shape[1]
    N = nullspace_modp(np.concatenate([T[0].T, T[1].T, T[2].T], axis=0) % P)
    odd = np.nonzero(H.par[n] == 1)[0]
    N = mm(N, nullspace_modp(N[odd] % P))
    for g in ('cyc', 'flip'):
        G = B.group_matrix(g, n)
        N = mm(N, nullspace_modp(mm((G.T - np.eye(d, dtype=np.int64)) % P, N)))
    A_all = np.zeros((0, N.shape[1]), dtype=np.int64)
    for cond in ('collinear', 'origin', 'FF collinear'):
        rows, _ = B.sketch_rows(n, BOUNDARY[cond][0], N.shape[1] + 24)
        A_all = np.concatenate([A_all, mm(rows, N)])
    Kh = nullspace_modp(A_all)                      # homogeneous kernel (dimension = free)
    print('homogeneous directions left:', Kh.shape[1])
    cf = mm(N, Kh)[:, 0]
    t0 = time.time()
    lims = {l: lim_tensors(H, sols[l], 2 * l) for l in range(1, L)}
    lim0 = lim_tensors(H, c0, n); limf = lim_tensors(H, cf, n)
    print('limits computed (%.1fs)' % (time.time() - t0))
    E1, E2, E3 = lims[1], lims[2], lims[3]
    # known part of Disc R^(4) from products:  - E1E3 - E2^2/2 + E1^2E2 - E1^4/4
    known = add((-1, disc_product([E1, E3])), (Fraction(-1, 2), disc_product([E2, E2])),
                (1, disc_product([E1, E1, E2])), (Fraction(-1, 4), disc_product([E1, E1, E1, E1])))
    lhs0 = high_delta(add((1, lim0[1]), (1, known)), L)
    lhsf = high_delta(limf[1], L)
    # also check SLL at lower loops (consistency of the identity and of our conventions)
    R2 = add((1, E2[1]), (Fraction(-1, 2), disc_product([E1, E1])))
    R3 = add((1, E3[1]), (-1, disc_product([E1, E2])), (Fraction(1, 3), disc_product([E1, E1, E1])))
    print('SLL check  L=2:', len(high_delta(R2, 2)) == 0, '  L=3:', len(high_delta(R3, 3)) == 0)
    keys = sorted(set(lhs0) | set(lhsf))
    print('SLL equations at L=4:', len(keys))
    # solve lhs0 + t lhsf = 0
    t = None; ok = True
    for k in keys:
        a, b = lhsf.get(k, 0), lhs0.get(k, 0)
        if a:
            tk = (-b) * pow(a, -1, P) % P
            if t is None: t = tk
            elif t != tk: ok = False
        elif b:
            ok = False
    print('consistent:', ok, ' t =', rat(t) if t is not None else None)
    c4 = (c0 + t * cf) % P
    np.save('E4_coeffs.npy', c4)
    yfree = B.__class__  # placeholder
    from bootstrap import Bootstrap
    Bc = Bootstrap(H)
    print('E^(4): y-free terms =', Bc.count_terms(c4, n, range(6)), ' (form factor F_3^(4) has 11,208)')
