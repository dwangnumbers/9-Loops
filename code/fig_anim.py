"""Frames for a rotating view of the Delta=0 surface (for the beamer animate package)."""
import figs, os
import matplotlib.pyplot as plt
plt.rcParams['savefig.bbox'] = 'standard'
import matplotlib; matplotlib.use('Agg')
N = 36
for k in range(N):
    figs.fig_delta0(elev=24 + 8 * __import__('math').sin(2 * 3.14159 * k / N), azim=40 + 360 * k / N,
                    fname=os.path.join('anim', f'd0-{k}.png'), small=True)
print('done')
