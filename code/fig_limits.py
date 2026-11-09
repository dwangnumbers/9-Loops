"""E^(1) approaching its collinear-limit and origin asymptotics (function level, including zeta_2)."""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import spence
import figs   # style + colours
BLUE, ORANGE, INK, INK2 = figs.BLUE, figs.ORANGE, figs.INK, figs.INK2
z2 = np.pi**2 / 6
def E1(u, v, w): return spence(1/u) + spence(1/v) + spence(1/w)
def ope(T, S, F):
    den = F + F*S**2 + S*T + F**2*S*T + F*T**2
    return F/den, F*S**2/((1+T**2)*den), T**2/(1+T**2)
T = np.logspace(-6, -0.3, 60)
fig, ax = plt.subplots(figsize=(5.4, 3.3))
for (S, F, col, lab) in [(1.0, 1.0, BLUE, 'collinear: $T\\to0$ ($S=F=1$)'), (0.5, 2.0, figs.AQUA, 'collinear: $T\\to0$ ($S=1/2,\\ F=2$)')]:
    err = np.abs(E1(*ope(T, S, F)) - (-2*np.log(T)**2 - 2*np.log(S)**2 - 2*z2))
    ax.loglog(T, err, '-', color=col, label=lab)
u = np.logspace(-6, -0.3, 60)
err_o = np.abs(E1(u, u, u) - (-1.5*np.log(u)**2 - 3*z2))
ax.loglog(u, err_o, '-', color=ORANGE, label='origin: $\\hat u=\\hat v=\\hat w=x\\to0$')
ax.loglog(T, 0.15*T, ':', color=INK2, lw=1.4)
ax.text(3e-3, 0.15*3e-3*0.25, 'slope 1', color=INK2, fontsize=9, rotation=24)
ax.set_xlabel('$T$  (or $x$)')
ax.set_ylabel('$|\\mathcal{E}^{(1)}-\\mathrm{asymptotic\\ form}|$')
ax.legend(loc='lower right', fontsize=8.5)
fig.savefig(os.path.join(figs.OUT, 'limits.pdf')); plt.close(fig)
# print a few numbers for comparison
for t in (1e-2, 1e-4, 1e-6):
    print(t, E1(*ope(t, 1.0, 1.0)), -2*np.log(t)**2 - 2*z2)
