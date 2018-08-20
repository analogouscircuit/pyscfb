import sys
import pickle
import scfb
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import scfbutils

# import wav file
if len(sys.argv)==2:
    f_name = str(sys.argv[1])
    f_s, in_sig = scipy.io.wavfile.read(f_name)
    in_sig = np.array(in_sig, dtype=np.float32)
    in_sig = in_sig/(2**15)
    in_sig = in_sig/np.max(in_sig)
# if no data give, generate first 8 notes of Die Kunst der Fuge
else:   
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

## Model Parameters
# Peripheral parameters
num_peri_units = 100
peri_f_lo = 100.
peri_f_hi = 4000.
# Template parameters
num_templates = 100
temp_f_lo = 50.0
temp_f_hi = 4000.0
temp_num_h = 5       # number of template bumps
sigma = 0.02    # bump width parameters
mu = 0.1        # adaptation rate
# Other parameters
sig_len_n = len(in_sig)

## process through SCFB
peri = scfb.SCFB(peri_f_lo, peri_f_hi, num_peri_units, f_s)
peri.process_signal(in_sig, verbose=True)

## process through templates 
chunks = peri.chunks    # probably pickle this
freqs = np.logspace(np.log10(temp_f_lo), np.log10(temp_f_hi), num_templates)
temp_array = scfb.TemplateArray(chunks, sig_len_n, freqs, temp_num_h, sigma, mu)
temp_array.adapt(verbose=True)

## process through WTAN
templates = temp_array.templates    # probably pickle this
t = np.arange(len(templates[0].strengths))*(1./44100)
strengths = [t.strengths for t in templates]
strengths = np.ascontiguousarray(np.flipud(np.stack(strengths, axis=0)))
k = np.ones((strengths.shape[0], strengths.shape[0]))*20. # inhibition constant
# more elaborate inhibition schemes commented out below
# max_val = strengths.shape[0]*strengths.shape[1]
# for i in range(strengths.shape[0]):
#     for j in range(strengths.shape[0]):
#         k[i][j] = max_val/(max_val*(1 + abs(i-j))) + 2.
# k *= 5
tau = np.ones(strengths.shape[0])*0.001     # time constant for WTAN network
print("Running Winner-Take-All Network...")
wta_out = scfbutils.wta_net(strengths, k, strengths.shape[0], strengths.shape[1],
        1./44100, tau, 1., 2., 1.2)     # pickle this as well
print("Finished WTAN calculations!")

pickle.dump((chunks, templates, wta_out), open('batch_data_for_animation.pk',
    'wb'))
