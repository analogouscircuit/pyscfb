import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import scfb
import scfbutils

## Import data
templates, freqs = pickle.load(open('template_data.pkl', 'rb'))
sig_len = len(templates[0].f_vals)
times = np.arange(sig_len)/44100.


## Run WTAN
strengths = [t.strengths for t in templates]
# strengths = np.ascontiguousarray(np.flipud(np.stack(strengths, axis=0)))
strengths = np.ascontiguousarray(np.stack(strengths, axis=0))
k = np.ones((strengths.shape[0], strengths.shape[0]))*5. # inhibition constant
# more elaborate inhibition schemes commented out below
# max_val = strengths.shape[0]*strengths.shape[1]
# for i in range(strengths.shape[0]):
#     for j in range(strengths.shape[0]):
#         k[i][j] = max_val/(max_val*(1 + abs(i-j))) + 2.
# k *= 5
tau = np.ones(strengths.shape[0])*0.02     # time constant for WTAN network
M = 10.      # max spike rate (original: 1.)
N = 2.      # slope
sigma = 1.2*M # half-max point (original: 1.2)
print("Running Winner-Take-All Network...")
wta_out = scfbutils.wta_net(strengths, k, strengths.shape[0], strengths.shape[1],
        1./44100, tau, M, N, sigma)     # last 3: M, N, sigma for Naka-Rushton
print("Finished WTAN calculations!")


## Animation
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.set_xscale('log')

lines = []
patches = []
for k, t in enumerate(templates):
    ln, = ax1.plot([t.f_vals[0], t.f_vals[0]], [0, t.strengths[0]], color='k')
    lines.append(ln)
    # circle = Circle((t.f_vals[0], t.strengths[0]), wta_out[k][0])
    # patches.append(circle)
    ln, = ax1.plot([t.f_vals[0], t.f_vals[0]], [30., 30.-wta_out[k][0]],
            color='r')
    lines.append(ln)
    # ax2.plot(times, wta_out[k])
# p = PatchCollection(patches)
# ax1.add_collection(p)

def init():
    ax1.set_xlim(20.0, 4000.0)
    ax1.set_ylim(0.0, 30.0)
    return lines,

def update(k):
    for n in range(len(templates)):
        lines[2*n].set_data([templates[n].f_vals[k], templates[n].f_vals[k]],
                [0, templates[n].strengths[k]])
        # patches[n].set_radius(10*wta_out[n][k])
        lines[2*n+1].set_data([templates[n].f_vals[k], templates[n].f_vals[k]],
                [30.0, 30.-wta_out[n][k]])
    return lines,
    
ani = FuncAnimation(fig, update, frames=np.arange(0, sig_len, 100, dtype=int),
        init_func=init, interval=.0001, repeat=False)

plt.show()
