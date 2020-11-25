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
mt_h = 0.5
len_t = 0.150
len_n = int(fs*len_t)
idcs = np.arange(len_n, dtype=np.int32)

chunks = []
for p in range(1, num_h+1):
    factor = 1.05 if p == mt_h else 1.0
    print(f0*p*factor)
    chunks.append((idcs, np.ones(len_n)*f0*p*factor))

f_vals = np.ones(1)*f0*0.95
ta = scfb.TemplateArray(chunks, len_n, f_vals, 8, sigma, mu, scale, beta)
ta.adapt(verbose=True)
print(ta.templates[0].f_vals[-1])
phi = ta.templates[0].f_vals
s = ta.templates[0].strengths

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_xlim(100.0, 1500.0)
ax1.set_ylim(-.5, 1.25)
ax2.set_xlim(100.0, 1500.0)
ax2.set_ylim(-.5, 1.25)

ax1.set_ylabel("Magnitude", fontsize=16)
ax1.set_xlabel("Frequency (Hz)", fontsize=16)
ax2.set_ylabel("Magnitude", fontsize=16)
ax2.set_xlabel("Frequency (Hz)", fontsize=16)
ax1.set_title("Before Adaptation", fontsize=18)
ax2.set_title("After Adaptation", fontsize=18)
ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)

f_range = np.arange(20.0, 4000.0, 0.5)

t_lw = 1.5
d_lw = 1.5
i_lw = 1.0

for p in range(1, num_h+1):
    factor = 1.05 if p == mt_h else 1.0
    f_val = f0*p*factor
    ax1.plot([f_val, f_val], [0.0, 1.0], color='k',
            linewidth=i_lw, linestyle='-.')
    ax2.plot([f_val, f_val], [0.0, 1.0], color='k',
            linewidth=i_lw, linestyle='-.')
    ax1.plot([f_range[0], f_range[-1]], [0.0, 0.0], color='k', linewidth=i_lw,
            linestyle='-.')
    ax2.plot([f_range[0], f_range[-1]], [0.0, 0.0], color='k', linewidth=i_lw,
            linestyle='-.')

ax1.plot(f_range, scu.template_vals(f_range, phi[0], sigma, num_h, scale, beta),
        linewidth=t_lw)
ax1.plot(f_range, scu.template_dvals(f_range, phi[0], sigma, num_h, 3., beta),
        linewidth=d_lw)

ax2.plot(f_range, scu.template_vals(f_range, phi[-1], sigma, num_h, scale,
    beta), linewidth=t_lw)
ax2.plot(f_range, scu.template_dvals(f_range, phi[-1], sigma, num_h, 3., beta),
        linewidth=d_lw)

plt.show()
