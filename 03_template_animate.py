import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scfb

templates = pickle.load(open('template_data.pkl', 'rb'))
sig_len = len(templates[0].f_vals)
fig, ax = plt.subplots()
ax.set_xscale('log')
lines = []
for t in templates:
    ln, = plt.plot([t.f_vals[0], t.f_vals[0]], [0, t.strengths[0]])
    lines.append(ln)

def init():
    ax.set_xlim(20.0, 4000.0)
    ax.set_ylim(0.0, 30.0)
    return lines,

def update(k):
    for n in range(len(templates)):
        lines[n].set_data([templates[n].f_vals[k], templates[n].f_vals[k]],
                [0, templates[n].strengths[k]])
    return lines,
    
ani = FuncAnimation(fig, update, frames=np.arange(0, sig_len, 20, dtype=int),
        init_func=init, interval=.0001, repeat=False)

plt.show()
