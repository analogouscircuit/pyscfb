import numpy as np
import gammatone.filters as gtf
import matplotlib.pyplot as plt

# make signal
f_s = 44100
dt = 1/f_s
dur = 0.1
t = np.arange(0, dur, dt)
num_h = 8
f0 = 220.0
in_sig = np.zeros_like(t)
for p in range(1, num_h+1):
    in_sig += np.cos(2*np.pi*f0*p*t)

# set up gammatone stuff
erb_freqs = gtf.erb_space(100.0, 4000.0, num=100)
print(erb_freqs)
erb_coefs = gtf.make_erb_filters(f_s, erb_freqs)
filted = gtf.erb_filterbank(in_sig, erb_coefs)

for k, channel in enumerate(filted):
    plt.plot(t, channel+100-k, color='k')

plt.show()
