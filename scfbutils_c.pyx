import numpy as np
import scipy.signal as dsp
cimport numpy as np
cimport cython
from libc.math cimport cos, sin, fabs, log


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple process_data(np.ndarray[np.float64_t] in_sig, double f_c, double bw, double f_s,
        np.ndarray[np.float64_t] b_lpf, np.ndarray[np.float64_t] a_lpf, double scale_fac,
        double min_e, double eps, double k_i, double k_p):
    '''
    Function that actually performs the adaptive filtering. Generates a
    list of 3-element tuples (f0, times, output). All of these are needed
    to initiate the PLL stage.

    TODO: convert u (control signal) to a circular buffer.
    '''
    cdef int filt_ord = 2
    cdef int buf_size = filt_ord + 1
    # Allocate memory and circular buffer indices
    in_sig = np.concatenate((np.zeros(buf_size - 1), in_sig))
    cdef:
        double dt = 1.0/f_s
        double f_c_base = f_c
        int offset = filt_ord   # offset for adjusting index where necessary
        np.ndarray[np.float64_t] f_record = np.zeros_like(in_sig)
        np.ndarray[np.float64_t] err = np.zeros_like(in_sig)
        np.ndarray[np.float64_t] u = np.zeros(buf_size, dtype=np.float64)
        np.ndarray[np.float64_t] out_l = np.zeros(buf_size, dtype=np.float64)
        np.ndarray[np.float64_t] out_c = np.zeros_like(in_sig)
        np.ndarray[np.float64_t] out_u = np.zeros(buf_size, dtype=np.float64)
        np.ndarray[np.float64_t] env_l = np.zeros(buf_size, dtype=np.float64)
        np.ndarray[np.float64_t] env_c = np.zeros(buf_size, dtype=np.float64)
        np.ndarray[np.float64_t] env_u = np.zeros(buf_size, dtype=np.float64)
        np.ndarray[np.float64_t] b_l = np.zeros(filt_ord+1)
        np.ndarray[np.float64_t] b_c = np.zeros(filt_ord+1)
        np.ndarray[np.float64_t] b_u = np.zeros(filt_ord+1)
        np.ndarray[np.float64_t] a_l = np.zeros(filt_ord+1)
        np.ndarray[np.float64_t] a_c = np.zeros(filt_ord+1)
        np.ndarray[np.float64_t] a_u = np.zeros(filt_ord+1)
        unsigned int idx0 = 2
        unsigned int idx1 = 1
        unsigned int idx2 = 0
        int k
        int sig_len = len(in_sig)
        np.ndarray[np.int32_t] on_record = np.zeros(sig_len-buf_size+1, dtype=np.int32) 
        np.ndarray[np.int32_t] off_record = np.zeros(sig_len-buf_size+1, dtype=np.int32) 
        int num_on = 0
        int is_locked = 0   # assume FDL is not tracking initially
        

    for k in range(buf_size-1, sig_len):
        # Update coefficients
        bp_narrow_coefs(f_c - bw, bw, f_s, b_l, a_l)
        bp_narrow_coefs(f_c, bw, f_s, b_c, a_c)
        bp_narrow_coefs(f_c + bw, bw, f_s, b_u, a_u)

        # Update circular indices
        idx0 = (idx0 + 1)%buf_size
        idx1 = (idx1 + 1)%buf_size
        idx2 = (idx2 + 1)%buf_size
        
        # calculate output of each filter in triplet
        out_l[idx0] = in_sig[k]*b_l[0] + in_sig[k-1]*b_l[1] + in_sig[k-2]*b_l[2] -\
                      out_l[idx1]*a_l[1] - out_l[idx2]*a_l[2]
        out_c[k] = in_sig[k]*b_c[0] + in_sig[k-1]*b_c[1] + in_sig[k-2]*b_c[2] -\
                   out_c[k-1]*a_c[1] - out_c[k-2]*a_c[2]
        out_u[idx0] = in_sig[k]*b_u[0] + in_sig[k-1]*b_u[1] + in_sig[k-2]*b_u[2] -\
                      out_u[idx1]*a_u[1] - out_u[idx2]*a_u[2]

        # Now run outputs of filters through envelope detectors
        # (i.e. rectify & LPF).
        env_l[idx0] = (b_lpf[0]*fabs(out_l[idx0]) 
                     + b_lpf[1]*fabs(out_l[idx1]) 
                     + b_lpf[2]*fabs(out_l[idx2])
                     - a_lpf[1]*env_l[idx1]
                     - a_lpf[2]*env_l[idx2])
        env_c[idx0] = (b_lpf[0]*fabs(out_c[k])
                     + b_lpf[1]*fabs(out_c[k-1])
                     + b_lpf[2]*fabs(out_c[k-2])
                     - a_lpf[1]*env_c[idx1]
                     - a_lpf[2]*env_c[idx2])
        env_u[idx0] = (b_lpf[0]*fabs(out_u[idx0])
                     + b_lpf[1]*fabs(out_u[idx1])
                     + b_lpf[2]*fabs(out_u[idx2])
                     - a_lpf[1]*env_u[idx1]
                     - a_lpf[2]*env_u[idx2])

        # Check if tracking/locking condition is met
        if env_c[idx0] > min_e:  
                    # locking before filters are warmed up
            env_diff = ((env_u[idx0]*scale_fac)/env_c[idx0]) - \
                        (env_l[idx0]/env_c[idx0]) 
            if env_diff < eps: 
                if is_locked == 0:
                    is_locked = 1
                    on_record[num_on] = k - offset
                    num_on += 1
            else:
                if is_locked == 1:
                    is_locked = 0
                    off_record[num_on-1] = k - offset
        else:
            if is_locked == 1:
                is_locked = 0
                off_record[num_on-1] = k - offset

        # Calculate the error, control equations, frequency update
        # scale factor inserted here to avoid messing with dynamics
        err[k] = log(scale_fac*env_u[idx0]) - log(env_l[idx0])    
        u[idx0] = u[idx1] + k_p*(err[k]-err[k-1]) + dt*k_i*err[k]
        f_c = f_c_base + u[idx0]
        f_record[k] = f_c

    return (out_c[offset:sig_len], f_record[offset:sig_len], on_record, off_record, num_on)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bp_narrow_coefs(double f_c, double bw, double f_s, np.ndarray[np.float64_t] b, np.ndarray[np.float64_t] a):
    '''
    From Steven W. Smith's book, The Scientist and Engineer's Guide to Digital 
    Signal Processing, eqs. 19-7 and 19-8. Note that his coefficient 
    conventions are different from Julius O. Smith's: a is b and b is a, and 
    the signs of the a coefficients (in JOS notation) are flipped.
    '''
    cdef double pi = np.pi
    cdef double f  = f_c/f_s
    bw = bw/f_s
    cdef double R = 1.0 - 3*bw
    cdef double K = (1.0 - 2*R*cos(2.0*pi*f) + (R*R))/(2.0 - 2.0*cos(2.0*pi*f))
    b[0] = 1.0 - K
    b[1] = 2*(K-R)*cos(2*pi*f)
    b[2] = (R*R)-K
    a[1] = -2*R*cos(2*pi*f)
    a[2] = (R*R)




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray pll(np.ndarray[np.float64_t, ndim=1] in_sig, double f_c, double f_s):
    '''
    Implementation of a digital phase-locked loop (PLL) as presented in
    Sethares' "Rhythm and Transforms" book (2007).  Uses a gradient-ascent
    approach, correlating the input signal with an internal oscillator,
    adjusting the phase of the internal oscillator to maximize the correlation.
    The frequency estimate is found by calculating the gradient of the phase.
    '''
    cdef double pi = np.pi
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
    in_sig = np.concatenate( (np.zeros(buf_size-1, dtype=np.float64), in_sig)) 
    cdef int sig_len = len(in_sig)
    cdef np.ndarray[np.float64_t, ndim=1] phi = np.zeros_like(in_sig)
    cdef np.ndarray[np.float64_t, ndim=1] out = np.zeros_like(in_sig)
    cdef np.ndarray[np.float64_t, ndim=1] mod_sig = np.zeros(buf_size, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] lp_out = np.zeros(buf_size, dtype=np.float64)
    cdef unsigned int idx0 = 2
    cdef unsigned int idx1 = 1
    cdef unsigned int idx2 = 0
    cdef int k
    cdef double t = 0
    for k in range(buf_size - 1, sig_len):
        t = (k - buf_size + 1)*dt
        idx0 = (idx0 + 1)%buf_size
        idx1 = (idx1 + 1)%buf_size
        idx2 = (idx2 + 1)%buf_size
        mod_sig[idx0] = in_sig[k]*sin(2*pi*f_c*t + phi[k-1]) 
        lp_out[idx0] = b_lp[0]*mod_sig[idx0] + b_lp[1]*mod_sig[idx1] \
                     + b_lp[2]*mod_sig[idx2] - a_lp[1]*lp_out[idx1] \
                     - a_lp[2]*lp_out[idx2]
        # lp_out[idx0] = b_lp0*mod_sig[idx0] + b_lp1*mod_sig[idx1] \
        #              + b_lp2*mod_sig[idx2] - a_lp1*lp_out[idx1] \
        #              - a_lp2*lp_out[idx2]
        phi[k] = phi[k-1] - mu*lp_out[idx0]
        out[k] = cos(2*pi*f_c*t + phi[k])
    cdef np.ndarray freq_offset = np.gradient(phi[buf_size-1:sig_len], dt)/(2*pi)
    freq_offset = dsp.filtfilt(b_s, a_s, freq_offset)   #smooth freq estimates
    # return out[buf_size-1:], (freq_offset+f_c)
    return freq_offset+f_c

################################################################################


