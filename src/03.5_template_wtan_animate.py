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
ax1 = fig.add_subplot(1,1,1)
lines = []


def init():
    ax1.set_xscale('log')
    ax1.set_xlim(20.0, 4000.0)
    ax1.set_ylim(0.0, 40.0)
    ax1.set_xlabel("Frequency (Hz)", size=16)
    ax1.set_ylabel("Strength", size=16)
    for k, t in enumerate(templates):
        ln, = ax1.plot([t.f_vals[0], t.f_vals[0]], [0, t.strengths[0]], color='k')
        lines.append(ln)
        ln, = ax1.plot([t.f_vals[0], t.f_vals[0]], [30., 30.-wta_out[k][0]],
                color='r')
        lines.append(ln)
    text = ax1.text(25, 36, "$time$ = {:.2f}".format(0), fontsize=16)
    return lines, text

def update(k):
    for n in range(len(templates)):
        lines[2*n].set_data([templates[n].f_vals[k], templates[n].f_vals[k]],
                [0, templates[n].strengths[k]])
        # patches[n].set_radius(10*wta_out[n][k])
        lines[2*n+1].set_data([templates[n].f_vals[k], templates[n].f_vals[k]],
                [40.0, 40.-0.25*wta_out[n][k]])
        text.set_text("$time$ = {:.2f}".format(k/44100))
    return lines, text
    
ani = FuncAnimation(fig, update, frames=np.arange(0, sig_len, 100, dtype=int),
        init_func=init, interval=20, repeat=False)

# ani = FuncAnimation(fig, update, frames=np.arange(int(1.65*44100),
#     int(1.8*44100), 50, dtype=int),
#         init_func=init, interval=20, repeat=False)

# ani = FuncAnimation(fig, update, frames=np.array(
#     [1.68*44100, 1.70*44100, 1.72*44100, 1.74*44100
#         ], dtype=int),
#         init_func=init, interval=900, repeat=False)

# ani = FuncAnimation(fig, update, [79380],
#         init_func=init, interval=.0001, repeat=False)

plt.show()
