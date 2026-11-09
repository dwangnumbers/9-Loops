"""Which adjacent letter pairs occur in the extended-Steinmann hexagon space?  (support of the 40-dim pair space)"""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from hexspace import HexSpace, P, nullspace_modp
from bootstrap import matmul_modp as mm
import figs
H = HexSpace('ext', verbose=False).build(8)
n = 8
# pair content of the slots (n-1, n) over ALL basis elements of H_n:
# tail of basis element i on lower basis k:  V[(x,y)] = T_n[y][i,:] @ T_{n-1}[x][:, k]
cols = []
for x in range(9):
    for y in range(9):
        cols.append(mm(H.T[n][y], H.T[n - 1][x]).reshape(-1))
V = np.array(cols)                                  # 81 x (d_n d_{n-2})
R = np.random.default_rng(1).integers(0, P, (V.shape[1], 100))      # random sketch of the column space
rank = 81 - nullspace_modp(mm(V % P, R).T.copy()).shape[1]
support = np.any(V != 0, axis=1).reshape(9, 9)
print('dimension of the pair space in the last two slots of H_8:', rank)
print('pairs that never occur:', [(int(a), int(b)) for a, b in zip(*np.nonzero(~support))])
names = [r'$\hat a$', r'$\hat b$', r'$\hat c$', r'$\hat d$', r'$\hat e$', r'$\hat f$', r'$y_u$', r'$y_v$', r'$y_w$']
fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.imshow(support, cmap=ListedColormap(['#e34948', '#cde2fb']), vmin=0, vmax=1)
for i in range(9):
    for j in range(9):
        if not support[i, j]:
            ax.text(j, i, '×', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
ax.set_xticks(range(9)); ax.set_yticks(range(9))
ax.set_xticklabels(names, fontsize=9); ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('next letter $x_{k+1}$'); ax.set_ylabel('letter $x_k$')
ax.grid(False)
ax.set_xticks(np.arange(-.5, 9, 1), minor=True); ax.set_yticks(np.arange(-.5, 9, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.5); ax.tick_params(which='minor', length=0)
fig.savefig(os.path.join(figs.OUT, 'pairs.pdf')); plt.close(fig)
