import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp

def PLL(in_sig, f_c, f_s, mu=0.03):
    dt = 1./f_s
    lp_cutoff = f_c/12.0
    b_lp, a_lp = dsp.bessel(2, (lp_cutoff*2)/f_s)
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
    return out[buf_size-1:], (freq_offset+f_c)

##############################################################################
if __name__ == "__main__":
    f_s = 44100
    dt = 1./f_s
    f_in = 2300
    dur = 0.05
    times = np.arange(0, dur, dt)
    in_sig = np.cos(2*np.pi*f_in*times)
    out, freqs = PLL(in_sig, 2200.0, f_s)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(times, freqs)
    ax2.plot(times, out)
    ax2.plot(times, in_sig)
    plt.show()
