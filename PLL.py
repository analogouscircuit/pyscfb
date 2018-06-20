import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp


def PLL(in_sig, f_c, f_s):
    '''
    Implementation of a digital phase-locked loop (PLL) as presented in
    Sethares' "Rhythm and Transforms" book (2007).  Uses a gradient-ascent
    approach, correlating the input signal with an internal oscillator,
    adjusting the phase of the internal oscillator to maximize the correlation.
    The frequency estimate is found by calculating the gradient of the phase.
    '''
    mu = 0.120*(f_c/4000.)
    dt = 1./f_s
    lp_cutoff = f_c/5.0
    b_lp, a_lp = dsp.bessel(2, (lp_cutoff*2)/f_s)
    b_s, a_s = dsp.bessel(2, (50.0*2)/f_s)      # smoothing filter
    buf_size = max(len(b_lp), len(a_lp))
    in_sig = np.concatenate( (np.zeros(buf_size-1), in_sig)) 
    phi = np.zeros_like(in_sig)
    out = np.zeros_like(in_sig)
    mod_sig = np.zeros(buf_size, dtype=np.float32)
    lp_out = np.zeros(buf_size, dtype=np.float32)
    idx0 = 2
    idx1 = 1
    idx2 = 0
    for k in range(buf_size - 1, len(in_sig)):
        t = (k - buf_size + 1)*dt
        idx0 = (idx0 + 1)%buf_size
        idx1 = (idx1 + 1)%buf_size
        idx2 = (idx2 + 1)%buf_size
        mod_sig[idx0] = in_sig[k]*np.sin(2*np.pi*f_c*t + phi[k-1]) 
        lp_out[idx0] = b_lp[0]*mod_sig[idx0] + b_lp[1]*mod_sig[idx1] \
                     + b_lp[2]*mod_sig[idx2] - a_lp[1]*lp_out[idx1] \
                     - a_lp[2]*lp_out[idx2]
        phi[k] = phi[k-1] - mu*lp_out[idx0]
        out[k] = np.cos(2*np.pi*f_c*t + phi[k])
    freq_offset = np.gradient(phi[buf_size-1:], dt)/(2*np.pi)
    freq_offset = dsp.filtfilt(b_s, a_s, freq_offset)   #smooth freq estimates
    return out[buf_size-1:], (freq_offset+f_c)

##############################################################################
if __name__ == "__main__":
    f_s = 44100
    dt = 1./f_s
    f_c = 1800.0
    f_in = f_c*1.1
    print("PLL f_c: %.2f \nInput f: %.2f" %(f_c, f_in))
    dur = 0.55
    times = np.arange(0, dur, dt)
    in_sig = np.cos(2*np.pi*f_in*times)
    out, freqs = PLL(in_sig, f_c, f_s)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(times, freqs, f_c)
    ax2.plot(times, out)
    ax2.plot(times, in_sig)
    plt.show()
