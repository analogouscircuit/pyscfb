import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scfb
import scfbutils as scu
from matplotlib.animation import FuncAnimation

sigma = 0.03
mu = 1.
fs = 44100

# generate input (harmonic complex with mistuned 3rd partial)
f0 = 200.0
# heights = np.array([(1.5/p)**.9 for p in range(1, 7)])
heights = np.array([0.8, 1.0, 1.0, 1.0, 0.7, 0.5])
heights_s = np.array([(1/(p+1))**(0.5) for p in range(0,8)])
mt_h = 0.5
len_t = 0.010
# len_n = int(fs*len_t)
len_n = 80 
idcs = np.arange(len_n, dtype=np.int32)
print("Length of idcs: ", len(idcs))
num_h = 6
input_freqs = [np.zeros(1) for k in range(num_h)] 
stimuli = [np.array([1,2,3,4,5,6])*205.,
           np.array([1,2,3,4,5,6])*195.,
           np.array([-1,2,3,4,5,6])*200., # -1 just makes the input irrelevant
           np.array([1,2,3,4,5,6])*405.,
           np.array([1,2,3,4,5,6])*95.,
           np.array([1,2,3,4,5,6])*200.]

f_est = np.array([0])
s_vals = np.array([0])
d_vals = np.array([0])

f_vals = np.ones(1)*f0
for stimulus in stimuli:
    chunks = []
    for k, f in enumerate(stimulus): 
        input_freqs[k] = np.concatenate([input_freqs[k], np.ones(len_n)*f])
        chunks.append((idcs, np.ones(len_n)*f))
    ta = scfb.TemplateArray(chunks, len_n, f_vals, sigma, heights, mu)
    ta.adapt(verbose=True)
    print(ta.templates[0].f_vals[-1])
    phi = ta.templates[0].f_vals
    s = ta.templates[0].strengths
    d = np.array([np.sum(scu.template_dvals(stimulus, f, 0.03, heights)) for f in
        phi])
    f_vals = np.ones(1)*phi[-1]
    f_est = np.concatenate([f_est, phi])
    s_vals = np.concatenate([s_vals, s])
    d_vals = np.concatenate([d_vals, d])

# do single partial moving
chunks = []
len_n *= 3
idcs = np.arange(len_n, dtype=np.int32)
for k, f in enumerate([200., 400., 600., 800., 1000., 1200.]):
    if k != 2:
        input_freqs[k] = np.concatenate([input_freqs[k], np.ones(len_n)*f])
        chunks.append((idcs, np.ones(len_n)*f))
    else:
        input_freqs[k] = np.concatenate([input_freqs[k], np.linspace(f, 1.1*f, len_n)])
        chunks.append((idcs, np.linspace(f, 1.1*f, len_n)))
ta = scfb.TemplateArray(chunks, len_n, f_vals, sigma, heights, mu)
ta.adapt(verbose=True)
print(ta.templates[0].f_vals[-1])
phi = ta.templates[0].f_vals
s = ta.templates[0].strengths
d = np.array([np.sum(scu.template_dvals(stimulus, f, 0.03, heights)) for f in
    phi])
f_vals = np.ones(1)*phi[-1]
f_est = np.concatenate([f_est, phi])
s_vals = np.concatenate([s_vals, s])
d_vals = np.concatenate([d_vals, d])

f_est = f_est[1:]
s_vals = s_vals[1:]
d_vals = d_vals[1:]

print(len(input_freqs), input_freqs[0].shape)
## Animation
fig = plt.figure()
gs = gridspec.GridSpec(1,2, width_ratios=[10, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# colors
orange = 'C1'
red = 'C3'
blue = 'C0'
grey = 'C7'
c_strengths = red
c_template = blue
c_derivative = orange
c_input = grey

lines = []
# for f_val in stimuli[0]: 
#     ln, = ax1.plot([f_val, f_val], [0.0, 1.2], color=c_input,
#             linewidth=1.2)
#     lines.append(ln)
for p in range(num_h):
    ln, = ax1.plot([input_freqs[p][0], input_freqs[p][0]], [0.0, 1.2],
            color=c_input, linewidth=1.2)
    lines.append(ln)

ln, = ax1.plot([-100, 4000], [0, 0], color=c_input, linewidth=1.2)
lines.append(ln)

f_range = np.arange(20.0, 4000.0, 0.5)
ln, = ax1.plot(f_range, scu.template_vals(f_range, f_est[0], sigma, heights_s),
        color=c_strengths, linewidth=1.0)
lines.append(ln)
ln, = ax1.plot(f_range, scu.template_vals(f_range, f_est[0], sigma, heights),
        color=c_template, linestyle='--', linewidth=0.8)
lines.append(ln)
ln, = ax1.plot(f_range, scu.template_dvals(f_range, f_est[0], sigma, heights),
        color=c_derivative)#, linestyle='-.')
lines.append(ln)

bars = ax2.bar([-1, 1], [d_vals[0], s_vals[0]], color=[c_derivative, c_strengths])

text = ax1.text(875, -0.35, "$f_0$ = {:.1f}".format(f_est[0]), size=18)
q = 0

def init():
    ax1.set_xlim(0.0, 2000.0)
    ax1.set_ylim(-.5, 1.5)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 6)
    ax2.get_xaxis().set_ticks([-1, 1])
    ax2.set_xticklabels(["adapt value", "strength"], rotation=45)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    return bars, lines, text

def update(k):
    # if k%len_n == 0:
    #     q = int(k/len_n) 
    #     for p, f_val in enumerate(stimuli[q]):
    #         lines[p].set_data([f_val, f_val], [0.0, 1.2])
    for p in range(num_h):
        lines[p].set_data([input_freqs[p][k], input_freqs[p][k]], [0.0, 1.2])
    lines[-3].set_data(f_range, scu.template_vals(f_range, f_est[k], sigma,
        heights_s)) 
    lines[-2].set_data(f_range, scu.template_vals(f_range, f_est[k], sigma,
        heights)) 
    lines[-1].set_data(f_range, scu.template_dvals(f_range, f_est[k], sigma,
        heights))
    bars[0].set_height(d_vals[k])
    bars[1].set_height(s_vals[k])
    text.set_text("$f_0$ = {:.1f}".format(f_est[k]))
    return bars, lines, text

ani = FuncAnimation(fig, update, frames=np.arange(0, len(input_freqs[0])-5, 1, dtype=int),
        init_func=init, blit=False, interval = 0.1, repeat=False)

plt.show()
