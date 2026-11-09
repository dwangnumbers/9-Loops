"""Fill-in during exact Gaussian elimination (mod p) of a real bootstrap system: the weight-8 even-parity step."""
import numpy as np, os, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hexspace import HexSpace, P
from alphabet import ODD
import figs

H = HexSpace('ext', verbose=False).build(7)
n = 8
Tm = H.T[n - 1]; d1, d2 = Tm.shape[1], Tm.shape[2]
D = H.relations_for(n)
colpar = (H.par[n - 1][:, None] ^ np.array(ODD)[None, :]).reshape(-1)
cols = np.nonzero(colpar == 0)[0]
blocks = []
for m in range(D.shape[0]):
    B = np.zeros((d2, d1, 9), dtype=np.int64)
    for a in range(9):
        for b in range(9):
            if D[m, a, b]:
                B[:, :, b] = (B[:, :, b] + D[m, a, b] * Tm[a].T) % P
    B = B.reshape(d2, d1 * 9)[:, cols]
    blocks.append(B[np.any(B != 0, axis=1)])
M = np.concatenate(blocks) % P
rows, ncols = M.shape
print('system', M.shape, 'nnz', np.count_nonzero(M), 'density %.3f' % (np.count_nonzero(M) / M.size))

def mulmod(a, b):   # exact elementwise a*b mod p for a, b < 2^31 (split b into 16-bit halves)
    return ((a * (b & 0xFFFF)) % P + ((a * (b >> 16)) % P << 16) % P) % P

def eliminate(M, order, snap_fracs=()):
    """Gauss-Jordan mod p with a given column order; returns nnz history and snapshots of the sparsity pattern."""
    A = M[:, order].copy(); rows, ncols = A.shape
    r, hist, snaps = 0, [np.count_nonzero(A)], {}
    for c in range(ncols):
        nzr = np.nonzero(A[r:, c])[0]
        if len(nzr) == 0:
            continue
        pr = r + nzr[0]
        if pr != r:
            A[[r, pr]] = A[[pr, r]]
        A[r] = (A[r] * pow(int(A[r, c]), -1, P)) % P
        others = np.nonzero(A[:, c])[0]; others = others[others != r]
        if len(others):
            f = (P - A[others, c]) % P
            nzc = np.nonzero(A[r])[0]
            upd = mulmod(np.repeat(f[:, None], len(nzc), axis=1), np.tile(A[r, nzc], (len(others), 1)))
            A[np.ix_(others, nzc)] = (A[np.ix_(others, nzc)] + upd) % P
        r += 1
        hist.append(np.count_nonzero(A))
        for fr in snap_fracs:
            if fr not in snaps and r >= fr * 735:
                snaps[fr] = A != 0
        if r == rows:
            break
    return np.array(hist), snaps, r

rng = np.random.default_rng(0)
nat = np.arange(M.shape[1])
colnnz = np.count_nonzero(M, axis=0)
orders = {'natural order': nat, 'random order': rng.permutation(M.shape[1]),
          'sparsest columns first': np.argsort(colnnz, kind='stable')}
res = {}
for name, od in orders.items():
    t0 = time.time()
    hist, snaps, rank = eliminate(M, od, snap_fracs=(0.0, 0.5, 1.0) if name == 'natural order' else ())
    res[name] = (hist, snaps)
    print(f'{name:24s} rank {rank}  kernel {M.shape[1]-rank}  peak nnz {hist.max():,}  final nnz {hist[-1]:,}  ({time.time()-t0:.1f}s)')

fig = plt.figure(figsize=(10.0, 3.3))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.6], wspace=0.55)
snaps = res['natural order'][1]
titles = {0.0: 'start (2.3% dense)', 0.5: 'half the pivots done', 1.0: 'reduced echelon form'}
first = {0.0: M[:, nat] != 0}
for k, fr in enumerate((0.0, 0.5, 1.0)):
    ax = fig.add_subplot(gs[0, k])
    pat = first[0.0] if fr == 0.0 else snaps[fr]
    ax.spy(pat, markersize=0.15, color=figs.BLUE, aspect='auto')
    ax.set_title(titles[fr], fontsize=9, color=figs.INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(figs.GRID)
ax = fig.add_subplot(gs[0, 3])
cols_ = {'natural order': figs.BLUE, 'random order': figs.ORANGE, 'sparsest columns first': figs.AQUA}
for name, (hist, _) in res.items():
    ax.plot(np.arange(len(hist)), hist / 1e3, color=cols_[name], label=name)
ax.set_xlabel('pivots eliminated'); ax.set_ylabel('nonzeros (thousands)')
ax.set_ylim(0, 285); ax.legend(fontsize=8, loc='upper right')
ax.set_title(f'{M.shape[0]}$\\times${M.shape[1]} system, kernel dim 282', fontsize=9, color=figs.INK)
fig.savefig(os.path.join(figs.OUT, 'fillin.pdf'), bbox_inches='tight'); plt.close(fig)
