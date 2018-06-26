import numpy as np
import scipy.signal as dsp
cimport numpy as np


def pll(np.ndarray[np.float64_t, ndim=1] in_sig, double f_c, double f_s):
    '''
    Implementation of a digital phase-locked loop (PLL) as presented in
    Sethares' "Rhythm and Transforms" book (2007).  Uses a gradient-ascent
    approach, correlating the input signal with an internal oscillator,
    adjusting the phase of the internal oscillator to maximize the correlation.
    The frequency estimate is found by calculating the gradient of the phase.
    '''
    cdef double mu = 0.120*(f_c/4000.)
    cdef double dt = 1./f_s
    cdef double lp_cutoff = f_c/5.0
    cdef double b_lp0, b_lp1, b_lp2, a_lp1, a_lp2
    cdef double b_s0, b_s1, b_s2, a_s1, a_s2
    cdef np.ndarray[np.float64_t] b_lp, a_lp, b_s, a_s
    b_lp, a_lp = dsp.bessel(2, (lp_cutoff*2)/f_s)
    b_s, a_s = dsp.bessel(2, (50.0*2)/f_s)      # smoothing filter
    # b_lp0, b_lp1, b_lp2 = b_lp[0], b_lp[1], b_lp[2]
    # a_lp1, a_lp2 = a_lp[1], a_lp[2]
    # b_s0, b_s1, b_s2 = b_s[0], b_s[1], b_s[2]
    # a_s1, a_s2 = a_s[1], a_s[2]
    cdef int buf_size = max(len(b_lp), len(a_lp))
    in_sig = np.concatenate( (np.zeros(buf_size-1, dtype=np.float32), in_sig)) 
    cdef int sig_len = len(in_sig)
    cdef np.ndarray[np.float64_t, ndim=1] phi = np.zeros_like(in_sig)
    cdef np.ndarray[np.float64_t, ndim=1] out = np.zeros_like(in_sig)
    cdef np.ndarray[np.float64_t, ndim=1] mod_sig = np.zeros(buf_size, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] lp_out = np.zeros(buf_size, dtype=np.float64)
    cdef int idx0 = 2
    cdef int idx1 = 1
    cdef int idx2 = 0
    cdef int k
    for k in range(buf_size - 1, sig_len):
        t = (k - buf_size + 1)*dt
        idx0 = (idx0 + 1)%buf_size
        idx1 = (idx1 + 1)%buf_size
        idx2 = (idx2 + 1)%buf_size
        mod_sig[idx0] = in_sig[k]*np.sin(2*np.pi*f_c*t + phi[k-1]) 
        lp_out[idx0] = b_lp[0]*mod_sig[idx0] + b_lp[1]*mod_sig[idx1] \
                     + b_lp[2]*mod_sig[idx2] - a_lp[1]*lp_out[idx1] \
                     - a_lp[2]*lp_out[idx2]
        # lp_out[idx0] = b_lp0*mod_sig[idx0] + b_lp1*mod_sig[idx1] \
        #              + b_lp2*mod_sig[idx2] - a_lp1*lp_out[idx1] \
        #              - a_lp2*lp_out[idx2]
        phi[k] = phi[k-1] - mu*lp_out[idx0]
        out[k] = np.cos(2*np.pi*f_c*t + phi[k])
    cdef np.ndarray freq_offset = np.gradient(phi[buf_size-1:], dt)/(2*np.pi)
    freq_offset = dsp.filtfilt(b_s, a_s, freq_offset)   #smooth freq estimates
    return out[buf_size-1:], (freq_offset+f_c)
