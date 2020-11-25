import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle

signal, f_s, chunks, templates, wta_out = pickle.load(open('batch_data_for_animation.pkl',
    'rb'))

t = np.arange(len(signal))/f_s

print("chunks shape: ", chunks[0])
print("template estimates shape: ", templates[0].f_vals.shape)
print("template strengths shape: ", templates[0].strengths.shape)
print("wta output shape: ", wta_out.shape)

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

# first plot chunks -- then modify to make incremental, animated plot
num_chunks = len(chunks)
# ax2.set_ylim(100.0, 4000.0)
# ax2.set_xlim(t[0], t[-1])
# for chunk in chunks:
#     ax2.plot(chunk[0]/f_s, chunk[1], color='r', linewidth=0.8)

scfb_lines = []

# set up initial lines
for chunk in chunks:
    ln, = ax2.plot(chunk[0][0]/f_s, chunk[1][0], color='r', linewidth=0.8)
    scfb_lines.append(ln)

def init():
    ax2.set_ylim(100.0, 4000.0)
    ax2.set_xlim(t[0], t[-1])
    return scfb_lines

def update(k):
    for n, chunk in enumerate(chunks):
        if k in chunk[0]:
            scfb_lines[n].set_data(chunk[0][0:k]/f_s, chunk[1][0:k])
    return scfb_lines

ani = FuncAnimation(fig, update, frames=np.arange(0, len(signal), 200,
    dtype=int), init_func=init, interval = 0.001, repeat=False)

plt.show()
