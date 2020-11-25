import sys
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import scfb
import scfbutils
import gammatone.filters as gtf

## generate two-tone signal
f_s = 44100
dt = 1/f_s
num_h = 1
# freqs = [300, 300.*(5/4), 300.*(3/2)]
freqs = [333, 500]
dur = 0.2
t = np.arange(dur*f_s)/f_s
in_sig = np.zeros_like(t)
for freq in freqs:
    for p in range(1, num_h+1):
        in_sig += ((1/p)**0.0)*np.cos(2*np.pi*freq*p*t) # +
                # np.random.random(1)*2*np.pi)
in_sig /= np.max(in_sig) 

## process through SCFB
peri = scfb.SCFB(100., 4000., 100, f_s, filt_type='gammatone', bounding=True)
peri.process_signal(in_sig, verbose=True)
sig_len_n = len(in_sig)

## process through templates
chunks = peri.chunks 
freqs = np.logspace(np.log10(100.0), np.log10(1600.0), 100)
bump_heights = [(1/k)**0.9 for k in range(1,7)]
bump_heights = np.array(bump_heights)
sigma = 0.03 
mu = 1
temp_array = scfb.TemplateArray(chunks, sig_len_n, freqs, sigma,
        bump_heights, mu)
temp_array.adapt(verbose=True)
templates = temp_array.templates
sig_len = len(templates[0].f_vals)
times = np.arange(sig_len)/44100.
CFs = freqs 
pitches = np.array([t.f_vals[-2] for t in templates])
strengths = np.array([t.strengths[-2] for t in templates])

## plot buttes
strengths /= np.max(strengths)  
idcs = strengths > 0.99
pitches *= idcs
fig = plt.figure()
ax1 = fig.add_subplot(1,2,2)
ax1.set_xscale("log")
ax1.stem(CFs, pitches, basefmt=" ")
skip = 10
ax1.set_xticks([cf for cf in CFs[::skip]])
ax1.grid("on", axis='y')
ax1.set_xlabel("CF of Adapative Template", size=16)
ax1.set_ylabel("Phase-locked firing rate (Hz)", size=16)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.tick_params(axis='both', which='both', labelsize=12)

## place-rate profile, place-rate coding
f_c = gtf.erb_space(100., 1600., num=100)
erb_coefs = gtf.make_erb_filters(f_s, f_c)
filt_channels = gtf.erb_filterbank(in_sig, erb_coefs)
channel_power = np.zeros(len(filt_channels))
for k in range(len(channel_power)):
    channel_power[k] = np.sqrt(np.dot(filt_channels[k], filt_channels[k]))
channel_power /= np.max(channel_power)
ax2 = fig.add_subplot(1,2,1)
ax2.set_xscale("log")
skip = 10
ax2.set_xticks([cf for cf in f_c[::skip]])
ax2.stem(f_c, channel_power, basefmt=" ")
ax2.grid("on", axis='y')
ax2.set_xlabel("CF of Auditory Channel", size=16)
ax2.set_ylabel("Normalized Power/Firing Rate", size=16)
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.tick_params(axis='both', which='both', labelsize=12)

plt.show()
