import numpy as np
import matplotlib.pyplot as plt
import scfb
import scfbutils as scu
from matplotlib.animation import FuncAnimation

sigma = 0.03
mu = 1.
scale = 1.0
beta = 0.9

# generate input (harmonic complex with mistuned 3rd partial)
f0 = 205.0
fs = 44100
num_h = 6
heights = np.array([(1.5/p)**.9 for p in range(1, 7)])
mt_h = 0.5
len_t = 0.150
len_n = int(fs*len_t)
idcs = np.arange(len_n, dtype=np.int32)

chunks = []
for p in range(1, num_h+1):
    factor = 1.05 if p == mt_h else 1.0
    print(f0*p*factor)
    chunks.append((idcs, np.ones(len_n)*f0*p*factor))

f_vals = np.ones(1)*f0*1.05
ta = scfb.TemplateArray(chunks, len_n, f_vals, 8, sigma, mu, scale, beta)
ta.adapt(verbose=True)
print(ta.templates[0].f_vals[-1])
phi = ta.templates[0].f_vals
s = ta.templates[0].strengths

fig, ax = plt.subplots()

lines = []
for p in range(1, num_h+1):
    factor = 1.05 if p == mt_h else 1.0
    f_val = f0*p*factor
    ln, = plt.plot([f_val, f_val], [0.0, 1.2], linestyle='--', color='999999')
    lines.append(ln)

f_range = np.arange(20.0, 4000.0, 0.5)
ln, = plt.plot(f_range, scu.template_vals(f_range, phi[0], sigma, heights))
lines.append(ln)
ln, = plt.plot(f_range, scu.template_dvals(f_range, phi[0], sigma, heights), linestyle='-.')
lines.append(ln)

def init():
    ax.set_xlim(0.0, 1500.0)
    ax.set_ylim(-.5, 1.5)
    return lines,

def update(k):
    lines[-2].set_data(f_range, scu.template_vals(f_range, phi[k], sigma,
        heights)) 
    lines[-1].set_data(f_range, scu.template_dvals(f_range, phi[k], sigma,
        heights))
    return lines,

ani = FuncAnimation(fig, update, frames=np.arange(0, len_n, 1, dtype=int),
        init_func=init, interval = 0.1, repeat=False)

plt.show()
