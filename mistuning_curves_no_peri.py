import sys
import pickle
import scfb
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

## Individual Harmonics Mistuned -- experiment parameters
mistunings = np.arange(1.0, 1.08, 0.005)
h_mistuned = 3
num_h = 6
f0 = 440.0
fs = 44100.
sig_len_t = 0.150
sig_len_n = int(sig_len_t*fs)
t = np.arange(sig_len_n)/fs
pitch = np.zeros_like(mistunings) 
chunk_idcs = np.arange(sig_len_n, dtype=np.int32)
# template parameters
t_freqs = np.logspace(np.log10(f0), np.log10(f0), 1)

# Template parameters
sigma = 0.033
mu = 1
scale = 1.0
beta = 0.9

pitch_rel = np.zeros_like(mistunings)
for q in range(num_h):
    h_mistuned = q + 1
    for k in range(len(mistunings)):
        # make the input chunks
        chunks = []
        for p in range(1, num_h+1):
            if p == h_mistuned:
                chunks.append((chunk_idcs, np.ones(sig_len_n)*f0*p*mistunings[k]))
            else:
                chunks.append((chunk_idcs, np.ones(sig_len_n)*f0*p))
        if k == 0:
            temp_array = scfb.TemplateArray(chunks, sig_len_n, t_freqs, num_h,
                    sigma, mu, scale, beta)
        else:
            temp_array.new_data(chunks, sig_len_n)
        temp_array.adapt()

        strengths = [t.strengths[-1] for t in temp_array.templates]
        pitch_idx = np.argmax(strengths)
        pitch[k] = temp_array.templates[pitch_idx].f_vals[-1]
    pitch_rel += pitch/f0

pitch_rel /= num_h
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_ylabel("Perceived pitch shift (percentage)", size=14)
ax1.set_xlabel("Degree of partial mistuning (percentage)", size=14)
ax2.set_ylabel("Perceived pitch (Hz)",size=14)
ax2.set_xlabel("Lowest frequency as multiple of\nmissing fundamental frequency",
        size=14)
ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)

ax1.plot((mistunings-1)*100, (pitch_rel-1)*100., color='b')


# Shift Complex -- experiment parameters
mistunings = np.arange(2.5, 5.5, 0.05)
idcs1 = mistunings < 3.45
idcs3 = mistunings > 4.45
idcs2 = np.array(idcs1 + idcs3, dtype=int)
idcs2 = np.ones_like(idcs2,dtype=bool) - idcs2
idcs2 = np.array(idcs2, dtype=bool)
num_h = 12
f0 = 200
t_freqs = np.logspace(np.log10(f0), np.log10(f0), 1)

pitches = np.zeros_like(mistunings)
for k in range(len(mistunings)):
    chunks = []
    for p in range(2, num_h+2):
        freqs = np.ones(sig_len_n)*f0 + f0*mistunings[k]
        chunks.append((chunk_idcs, freqs))

    if k == 0:
        temp_array = scfb.TemplateArray(chunks, sig_len_n, t_freqs, 6,
                sigma, mu, scale, beta)
    else:
        temp_array.new_data(chunks, sig_len_n)
    temp_array.adapt()

    strengths = [t.strengths[-1] for t in temp_array.templates]
    pitch_idx = np.argmax(strengths)
    pitches[k] = temp_array.templates[pitch_idx].f_vals[-1]

ax2.plot(mistunings[idcs1], pitches[idcs1], color='b')
ax2.plot(mistunings[idcs2], pitches[idcs2], color='b')
ax2.plot(mistunings[idcs3], pitches[idcs3], color='b')
plt.show()
