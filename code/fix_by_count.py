"""Which value t of the last four-loop direction reproduces the published 11,208-term form factor?"""
import numpy as np, collections, time
from hexspace import HexSpace, P, nullspace_modp
from bootstrap import Bootstrap, matmul_modp as mm, rat
from bootstrap2 import Bootstrap2, BOUNDARY

L, n = 4, 8
H = HexSpace('ext', verbose=False).build(n)
B2 = Bootstrap2(H); B = Bootstrap(H)
c0, _, free = B2.run(L, conditions=('collinear', 'origin', 'FF collinear'), verbose=False)
T = H.T[n]; d = T.shape[1]
N = nullspace_modp(np.concatenate([T[0].T, T[1].T, T[2].T], axis=0) % P)
odd = np.nonzero(H.par[n] == 1)[0]; N = mm(N, nullspace_modp(N[odd] % P))
for g in ('cyc', 'flip'):
    G = B2.group_matrix(g, n); N = mm(N, nullspace_modp(mm((G.T - np.eye(d, dtype=np.int64)) % P, N)))
A_all = np.zeros((0, N.shape[1]), dtype=np.int64)
for cond in ('collinear', 'origin', 'FF collinear'):
    rows, _ = B2.sketch_rows(n, BOUNDARY[cond][0], N.shape[1] + 24); A_all = np.concatenate([A_all, mm(rows, N)])
cf = mm(N, nullspace_modp(A_all))[:, 0]
# all y-free word coefficients of c0 and cf by meet in the middle
bot = B.word_vectors(4, range(6)); Bm = np.array(list(bot.values())).T
def top(c):
    cur = {(): c % P}
    for w in range(n, 4, -1):
        nxt = {}
        for word, vec in cur.items():
            for a in range(6):
                v = mm(vec[None, :], H.T[w][a])[0]
                nxt[(a,) + word] = v
        cur = nxt
    keys = sorted(cur); return keys, np.array([cur[k] for k in keys])
k0, T0 = top(c0); kf, Tf = top(cf)
A = mm(T0, Bm).ravel(); Bv = mm(Tf, Bm).ravel()
nz = Bv != 0
cand = (-A[nz] % P) * np.array([pow(int(x), -1, P) for x in Bv[nz]], dtype=object) % P
cnt = collections.Counter(int(x) for x in cand)
print('words touched by the free direction:', int(nz.sum()))
for t, m in cnt.most_common(4):
    tot = int(np.count_nonzero((A + (t * Bv) % P) % P))
    print(f't = {rat(t)}  makes {m} coefficients vanish  -> y-free terms = {tot}')
print('t = 0 ->', int(np.count_nonzero(A)))
union = int(np.count_nonzero((A != 0) | (Bv != 0)))
print('generic t -> y-free terms =', union)
hits = [ (t, m) for t, m in cnt.items() if union - m == 11208 ]
print('t values giving exactly 11208:', [(str(rat(t)), m) for t, m in hits][:10], ' of', len(cnt), 'candidates')
