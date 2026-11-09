"""Figures for the README. Run from code/; writes PDFs/PNGs into figs/."""
import json, math, os, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.special import spence

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
    'mathtext.fontset': 'cm', 'savefig.bbox': 'tight', 'savefig.dpi': 200, 'lines.linewidth': 2,
})

# ----------------------------------------------------------------------------------------------
# 1. dimensions of the hexagon symbol spaces (our computation; identical to Table 1 of 1906.07116)
def fig_hexdims(ext_dims):
    first = [1, 3, 9, 26, 75, 218, 643]
    stein = [1, 3, 6, 13, 29, 63, 134]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    w = np.arange(len(ext_dims))
    ax.semilogy(range(len(first)), first, '-o', color=ORANGE, ms=5, label='first entry + integrability')
    ax.semilogy(range(len(stein)), stein, '-o', color=AQUA, ms=5, label='+ Steinmann (first two slots)')
    ax.semilogy(w, ext_dims, '-o', color=BLUE, ms=5, label='+ extended Steinmann (all slots)')
    ax.text(w[-1] - 0.2, ext_dims[-1] * 1.9, f'$\\dim\\mathcal{{H}}_{{{w[-1]}}}={ext_dims[-1]:,}$',
            color=INK, ha='right', fontsize=10)
    ax.set_xlabel('weight $n$  (= 2 $\\times$ loops)')
    ax.set_ylabel('dimension of symbol space')
    ax.set_xticks(w)
    ax.legend(loc='upper left', fontsize=8.5)
    fig.savefig(os.path.join(OUT, 'hexdims.pdf'))
    plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 2. growth of the number of terms in the symbol: form factor F_3^(L) (= amplitude on Delta = 0)
def fig_growth():
    L = np.arange(1, 9)
    n = np.array([6, 12, 636, 11208, 263880, 4916466, 92954568, 1671656292], dtype=float)
    n9 = 30024320034
    fit = np.polyfit(L[2:], np.log10(n[2:]), 1)
    pred9 = 10 ** np.polyval(fit, 9)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.semilogy(L, n, 'o', color=BLUE, ms=6, label='published (1–8 loops)', zorder=3)
    Lf = np.linspace(3, 9, 50)
    ax.semilogy(Lf, 10 ** np.polyval(fit, Lf), '--', color=INK2, lw=1.2,
                label=f'fit $L=3..8$: $\\times{10**fit[0]:.1f}$ per loop')
    ax.semilogy([9], [n9], 'D', color=ORANGE, ms=7, label='nine loops (2026)', zorder=4)
    ax.annotate(f'{n9:,} terms\n(fit from $L\\leq 8$ predicted ' + f'{pred9/1e10:.1f}' + r'$\times10^{10}$)', (9, n9), xytext=(2.6, 4e10),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    for x, y in zip(L[:3], n[:3]):
        ax.annotate(f'{int(y)}', (x, y), xytext=(4, -12), textcoords='offset points', fontsize=8, color=INK2)
    ax.set_xlabel('loop order $L$'); ax.set_ylabel('nonzero terms in symbol')
    ax.set_xticks(range(1, 10)); ax.set_ylim(1, 3e12)
    ax.legend(loc='lower right', fontsize=8.5)
    fig.savefig(os.path.join(OUT, 'growth.pdf')); plt.close(fig)
    return 10 ** fit[0], pred9

# ----------------------------------------------------------------------------------------------
# 3. the Delta = 0 surface inside the cube of cross ratios, coloured by the one-loop function
def E1(u, v, w):
    # E^(1) = sum Li2(1 - 1/u_i);  scipy: spence(z) = Li2(1 - z)
    return spence(1 / u) + spence(1 / v) + spence(1 / w)

def ff_to_amp(u, v):
    w = 1 - u - v
    return v * w / ((1 - v) * (1 - w)), u * w / ((1 - u) * (1 - w)), u * v / ((1 - u) * (1 - v))

def fig_delta0(elev=28, azim=40, fname='delta0.pdf', small=False):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    m = 160
    s = np.linspace(0.002, 0.998, m)
    U, V = np.meshgrid(s, s)
    mask = U + V < 0.998
    U, V = np.where(mask, U, np.nan), np.where(mask, V, np.nan)
    uh, vh, wh = ff_to_amp(U, V)
    val = E1(uh, vh, wh)
    qs = np.nanpercentile(val, np.linspace(0, 100, 11))
    norm = colors.BoundaryNorm(qs, 256)
    cmap = colors.LinearSegmentedColormap.from_list('seqblue', ['#cde2fb', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'])
    fc = cmap(norm(np.nan_to_num(val, nan=0)))
    fig = plt.figure(figsize=(4.6, 4.2) if not small else (3.4, 3.2))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.plot_surface(uh, vh, wh, facecolors=fc, rstride=2, cstride=2, linewidth=0, antialiased=True, shade=False)
    # images of special form-factor lines
    t = np.linspace(0.01, 0.49, 200)
    a, b, c = ff_to_amp(t, t)           # (u,u,1-2u)
    ax.plot(a, b, c, color=ORANGE, lw=2.2, zorder=10)
    ax.text(*ff_to_amp(0.12, 0.12), '  image of $(u,u,1-2u)$', fontsize=7.5, color=INK, zorder=11)
    for (x, y, z, lab) in [(0.25, 0.25, 0.25, ''), (0, 0, 1, '$(0,0,1)$')]:
        ax.scatter([x], [y], [z], color=INK, s=12, zorder=12)
        ax.text(x + 0.05, y - 0.12, z + 0.02, lab, fontsize=8, color=INK, zorder=12)
    ax.set_xlabel('$\\hat u$', labelpad=-6); ax.set_ylabel('$\\hat v$', labelpad=-6); ax.set_zlabel('$\\hat w$', labelpad=-6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=elev, azim=azim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis._axinfo['grid']['color'] = GRID
    if not small:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08)
        cb.set_label('$\\mathcal{E}^{(1)}$ on $\\Delta=0$ (deciles)', fontsize=9)
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        cb.ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(OUT, fname), dpi=150 if fname.endswith('png') else 200)
    plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 4. sketching: probability that a random sketch misses rank, F_q
def rank_mod_q(A, q):
    A = A.copy() % q; r = 0; rows, cols = A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = A[r] * pow(int(A[r, c]), -1, q) % q
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % q
        r += 1
        if r == rows: break
    return r

def fig_sketch(trials=3000, seed=3):
    rng = np.random.default_rng(seed)
    r, mrows, ncols = 12, 40, 30
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for q, col in [(2, ORANGE), (3, AQUA), (7, BLUE)]:
        # a fixed constraint matrix of rank r over F_q
        A = (rng.integers(0, q, (mrows, r)) @ rng.integers(0, q, (r, ncols))) % q
        while rank_mod_q(A, q) != r:
            A = (rng.integers(0, q, (mrows, r)) @ rng.integers(0, q, (r, ncols))) % q
        ss = np.arange(0, 7)
        mc = []
        for s in ss:
            fails = 0
            for _ in range(trials):
                S = rng.integers(0, q, (r + s, mrows))
                if rank_mod_q((S @ A) % q, q) < r:
                    fails += 1
            mc.append(fails / trials if fails else np.nan)
        exact = [1 - np.prod([1 - float(q) ** (-j) for j in range(s + 1, r + s + 1)]) for s in ss]
        ax.semilogy(ss, exact, '-', color=col, lw=1.6)
        ax.semilogy(ss, mc, 'o', color=col, ms=5)
        ax.text(ss[-1] + 0.15, exact[-1], f'$q={q}$', color=INK, va='center', fontsize=9)
    ax.set_xlabel('oversampling $s$  (sketch rows $=$ rank $+\\,s$)')
    ax.set_ylabel('P(sketch kernel too large)')
    ax.set_xlim(-0.2, 7.0)
    fig.savefig(os.path.join(OUT, 'sketch.pdf')); plt.close(fig)

# ----------------------------------------------------------------------------------------------
# 5. the nine-loop cascades (numbers from the Cosmic9 method notes)
def fig_cascade9():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), sharey=False)
    ff = [('ansatz $\\mathcal{C}_5\\otimes B_{13}$', 450468), ('dihedral coords', 75923),
          ('adjacency (sketched)', 319), ('branch cuts', 60), ('collinear', 44), ('OPE LL', 1), ('OPE NLL', 0)]
    amp = [('ansatz $424\\times5431$', 2302744), ('$\\Delta=0$ data + lift', 8), ('origin', 5),
           ('$T^2$ LL+NLL', 3), ('$T^2$ NNLL', 1), ('$T^2$ N$^3$LL', 0)]
    for ax, data, title, col in [(axes[0], ff, 'form factor, 9 loops', AQUA), (axes[1], amp, 'amplitude lift, 9 loops', BLUE)]:
        y = np.arange(len(data))[::-1]
        vals = [max(v, 0.6) for _, v in data]
        ax.barh(y, vals, color=col, height=0.6, log=True)
        for yi, (lab, v) in zip(y, data):
            ax.text(max(v, 0.6) * 1.3, yi, f'{v:,}', va='center', fontsize=9, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([l for l, _ in data], fontsize=9)
        ax.set_xlim(0.5, 3e8); ax.set_title(title, fontsize=10, color=INK)
        ax.grid(axis='y', visible=False); ax.set_xlabel('free parameters (log scale)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'cascade9.pdf')); plt.close(fig)

if __name__ == '__main__':
    ext = [1, 3, 6, 13, 26, 51, 98, 184, 340, 613, 1085, 1887, 3224]
    if os.path.exists('H13.log'):
        txt = open('H13.log').read()
        if 'dim H_13 = 5431' in txt:
            ext.append(5431)
    fig_hexdims(ext)
    print('growth factor per loop %.2f, predicted 9-loop %.3e' % fig_growth())
    fig_delta0()
    fig_sketch()
    fig_cascade9()
    print('done')
