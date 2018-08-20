import sys
import pickle
import scfb
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt


# import wav file
if len(sys.argv)==2:
    f_name = str(sys.argv[1])
    f_s, in_sig = scipy.io.wavfile.read(f_name)
    in_sig = np.array(in_sig, dtype=np.float32)
    in_sig = in_sig/(2**15)
    in_sig = in_sig/np.max(in_sig)
# else:     # linear sliding (dynamic pitch shift example)
#     f_s = 44100
#     dur = 0.5
#     t = np.arange(0,f_s*dur)/f_s
#     in_sig = np.zeros_like(t)
#     num_h = 5
#     f0 = 400.0
#     freqs = np.linspace(f0, f0*1.25, len(t))
#     for p in range(1, num_h+1):
#         in_sig += np.cos(2*np.pi*p*freqs*t)
else:   # first 8 notes of Die Kunst der Fuge
    def gen_env(a, d, r, sus_val, dur_n, fs):
        t_a = np.arange(a*fs)/fs
        t_d = np.arange(d*fs)/fs
        t_r = np.arange(r*fs)/fs
        t_s_n = dur_n - len(t_a) - len(t_d) - len(t_r) 
        assert dur >= 0
        gamma = np.log(2)/a
        seg_a = np.exp(gamma * t_a) - 1
        gamma = -np.log(sus_val)/d
        seg_d = np.exp(-gamma*t_d)
        gamma = np.log(sus_val+1)/r
        seg_r = (sus_val+1) - np.exp(gamma*t_r)
        seg_s = np.ones(t_s_n)*sus_val
        return np.concatenate([seg_a, seg_d, seg_s, seg_r])
    f_s = 44100
    dt = 1/f_s
    num_h = 10 
    d = 293.6648; a = 440.0; f = 349.2282; cs = 277.1826; e = 329.6276;
    dur = 0.100     # eighth note duration
    freqs = [d, a, f, d, cs, d, e, f]
    durs = [4*dur, 4*dur, 4*dur, 4*dur, 4*dur, 2*dur, 2*dur, 5*dur]
    notes = []
    for k, freq in enumerate(freqs):
        num_cycles = int(durs[k]*freq)
        t = np.arange(0, num_cycles/freq, dt)
        note = np.zeros_like(t)
        for n in range(num_h):
            note += ((1/(n+1))**0.1)*np.cos(2*np.pi*freq*(n+1)*t)
        note = note * gen_env(0.02, durs[k] - 0.045, 0.02, 0.65, len(note), f_s)
        notes.append(note)
    in_sig = np.concatenate(notes)
    in_sig /= np.max(in_sig)
    in_sig += np.random.normal(scale=np.sqrt(0.000000001 * f_s / 2), size=in_sig.shape)
    scipy.io.wavfile.write("dkdf.wav", f_s, in_sig)
    in_sig *= 4 


# process through SCFB
peri = scfb.SCFB(100., 4000., 100, f_s)
peri.process_signal(in_sig, verbose=True)
sig_len_n = len(in_sig)

# plot results
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(np.arange(len(in_sig))/f_s, in_sig)

for fdl in peri.fdl:
    ax2.plot(np.arange(0,len(in_sig))/f_s, fdl.f_record, linewidth=0.7,
            color='k', linestyle='--')
peri.plot_output(ax2)
plt.show()

# write ordered data to file (for template processing)
pickle.dump((peri.chunks, sig_len_n), open("chunks.pkl", "wb"))

