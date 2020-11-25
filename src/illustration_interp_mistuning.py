import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
import scipy.fftpack as fft
import gammatone.filters as gtf


################################################################################
# Functions
################################################################################
def template_strength(f_vals_in, f0, num_h, sigma):
    strength = 0
    for f in f_vals_in:
        for p in range(1, num_h+1):
            strength += ((1/p)**0.1)*np.exp( -(f-p*f0)**2 / ( 2*(sigma*p*f0)**2 ))
    return strength

def parabolic_interp(points, vals):
    if len(points) != 3 or len(vals) != 3:
        print("Input error.  Both arrays must have length 3.")
        return 0
    return lambda x : (vals[0]*(x-points[1])*(x-points[2])/((points[0]- 
                       points[1])*(points[0]-points[2])) + 
                       vals[1]*(x-points[2])*(x-points[0])/((points[1]- 
                       points[2])*(points[1]-points[0])) + 
                       vals[2]*(x-points[0])*(x-points[1])/((points[2]- 
                       points[0])*(points[2]-points[1])))

def make_template(f0, num_h, w_size, fs, sigma=0.03):
    df = fs/w_size
    freqs = np.arange(0, fs/2, df)
    assert len(freqs) == w_size/2
    template = np.zeros(int(w_size/2))
    for k, f in enumerate(freqs):
        for p in range(1, num_h+1):
            template[k] += ((1/p)**0.1)*np.exp( -(f-p*f0)**2 / ( 2*(sigma*p*f0)**2 ))
    return template

def make_templates(f0_vals, num_h, w_size, fs, sigma=0.03):
    templates = np.zeros((len(f0_vals), int(w_size/2)))
    for k, f0 in enumerate(f0_vals):
        templates[k,:] = make_template(f0, num_h, w_size, fs, sigma)
    return templates

def template_power(in_sig, fs, f_vals, num_h = 6, sigma=0.033):
    templates = make_templates(f_vals, num_h, len(in_sig), fs, sigma)
    spec = np.abs(fft.fft(in_sig))
    spec = spec[:len(spec)//2]
    spec /= np.max(spec)
    power = np.zeros_like(f_vals)
    for k in range(len(f_vals)):
        power[k] = np.dot(spec, templates[k])
    return power


################################################################################
## Basic Parameters
################################################################################

## Signal parameters
f_s = 44100
dt = 1./f_s
num_samps = 2**12
t = np.arange(0, 2**12)*dt
num_h = 12
mistuned_h = 4
mistuning = 1.04
f0 = 155.0
f0_perceived = 155.0*1.005

## SAC model parameters
w_size = 2**10
delays = np.arange(0,w_size//2)*dt
num_channels = 70

## Template parameters
templates_per_octave_sparse = 10
templates_per_octave_dense = 100
sigma = 0.030
num_h = 5

## Plotting parameters
color_true = 'C1'
color_estimated = 'C7'
style_estimated = '-.'
color_interp = 'k'
style_interp = '--'
width_interp = 0.8
fontsize_label = 16
fontsize_ticks = 12
f_lim = [145, 170]
t_lim = [1.0/f for f in f_lim]


################################################################################
## Generate signal and input frequencies for template
################################################################################

## Template input
input_vals = [f0*p for p in range(2,num_h+1)]
input_vals[mistuned_h-1] *= mistuning

## Signal for SAC approach
in_sig = np.zeros_like(t)
in_sig_mt = np.zeros_like(t)
for p in range(2, num_h+1):
    if p == mistuned_h:
        in_sig_mt += np.cos(2*np.pi*f0*p*t*mistuning)
    else:
        in_sig_mt += np.cos(2*np.pi*f0*p*t)
    in_sig += np.cos(2*np.pi*f0*p*t)
in_sig /= np.max(in_sig)
in_sig_mt /=np.max(in_sig_mt)


################################################################################
# SAC Peak Picking 
################################################################################
f_c = gtf.erb_space(20., 5000., num=num_channels)
erb_coefs = gtf.make_erb_filters(f_s, f_c)
filt_channels = gtf.erb_filterbank(in_sig, erb_coefs)
filt_channels_mt = gtf.erb_filterbank(in_sig_mt, erb_coefs)
ac_channels = np.zeros((num_channels, w_size))
summary_ac = np.zeros(w_size, dtype=float)
ac_channels_mt = np.zeros((num_channels, w_size))
summary_ac_mt = np.zeros(w_size, dtype=float)

for k in range(num_channels):
    ac_channels[k,:] = dsp.correlate(filt_channels[k, -w_size:], 
            filt_channels[k, -w_size:], mode='same')
    ac_channels_mt[k,:] = dsp.correlate(filt_channels_mt[k, -w_size:], 
            filt_channels_mt[k, -w_size:], mode='same')
    summary_ac += np.clip(ac_channels[k,:],1,None)  # half-wave rectify
    summary_ac_mt += np.clip(ac_channels_mt[k,:],0,None)  # half-wave rectify

summary_ac /= np.max(summary_ac)
summary_ac /= 0.685 # normalize wrt non-zero bump
summary_ac_mt /= np.max(summary_ac_mt)
summary_ac_mt /= 0.685 # normalize wrt non-zero bump
summary_ac = summary_ac[w_size//2:]
summary_ac_mt = summary_ac_mt[w_size//2:]
offset = 100
est_idx = np.argmax(summary_ac_mt[offset:]) + offset

plt.subplot(131)
# plt.plot(delays, summary_ac,
#         color='k', 
#         linestyle='--',
#         linewidth=0.5)
plt.plot(delays, summary_ac_mt)
plt.plot([1./f0_perceived, 1./f0_perceived], [-5,5],
        color=color_true)
plt.plot([delays[est_idx], delays[est_idx]], [-5, 5],
        color=color_estimated,
        linestyle=style_estimated)
print("SAC Peak Estimate: ", 1./delays[est_idx])
plt.xlim(t_lim)
plt.ylim([-0.1, 1.1])
plt.xlabel("Delay Time", size=fontsize_label)
plt.ylabel("Power", size=fontsize_label)
plt.tick_params(axis='both', which='both', labelsize=fontsize_ticks)


################################################################################
# Template power diagram
################################################################################
## Dense templates
plt.subplot(132)
f_vals = np.logspace(np.log10(100), np.log10(200), templates_per_octave_dense)
t_power = np.zeros_like(f_vals)
for k, f0 in enumerate(f_vals):
    t_power[k] = template_strength(input_vals, f0, num_h, sigma)
t_power /= np.max(t_power)
est_idx = np.argmax(t_power)
plt.plot(f_vals, t_power)
plt.plot([f0_perceived, f0_perceived], [-5, 5], color=color_true)
plt.plot([f_vals[est_idx], f_vals[est_idx]], [-5, 5], 
        color=color_estimated,
        linestyle=style_estimated)
plt.xlim(f_lim)
plt.ylim([-0.1, 1.1])
plt.xlabel("Template F0", size=fontsize_label)
plt.ylabel("Template Power", size=fontsize_label)
plt.tick_params(axis='both', which='both', labelsize=fontsize_ticks)

## Sparse templates
plt.subplot(133)
f_vals = np.logspace(np.log10(100), np.log10(200), templates_per_octave_sparse)
t_power = np.zeros_like(f_vals)
for k, f0 in enumerate(f_vals):
    t_power[k] = template_strength(input_vals, f0, num_h, sigma)
t_power /= np.max(t_power)
peak_idx = np.argmax(t_power)
q = parabolic_interp(f_vals[peak_idx-1:peak_idx+2],
        t_power[peak_idx-1:peak_idx+2])
# interp_points = np.arange(148., 164.5, 0.1)
interp_points = np.arange(f_vals[peak_idx-1], f_vals[peak_idx+1], 0.1)
interp_vals = [q(x) for x in interp_points]
est_idx = np.argmax(interp_vals)
plt.plot(f_vals, t_power)
plt.plot([f0_perceived, f0_perceived], [-5, 5], color=color_true)
plt.plot([interp_points[est_idx], interp_points[est_idx]], [-5, 5],
        color=color_estimated,
        linestyle=style_estimated)
plt.plot(interp_points, interp_vals, 
        color=color_interp, 
        linewidth=width_interp, 
        linestyle=style_interp)
plt.xlim(f_lim)
plt.ylim([-0.1, 1.1])
plt.xlabel("Template F0", size=fontsize_label)
plt.ylabel("Template Power", size=fontsize_label)
plt.tick_params(axis='both', which='both', labelsize=fontsize_ticks)

plt.show()
