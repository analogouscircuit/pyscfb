'''
This is the central file for Cython. In includes functions defined in Cython
proper (cdefs and cpdefs) as well as wrapper functions (defs) for functions
defined in scfbutils_c.c.  

The functions are primarily loop/calculation intensive signal processing
components of the SCFB system (e.g. adaptive filters, phase-locked loops, filter
coefficient calculations).
'''

#distutils: sources=["scfbutils_c.c"]
import numpy as np
import scipy.signal as dsp
cimport numpy as np
cimport cython
cimport scfbutils_d as scd
from libc.math cimport cos, sin, fabs, log, exp
from libc.stdlib cimport malloc, free


################################################################################
# Extension Types
################################################################################

cdef class TemplateData:
    '''
    This class generates the linked list used by the Template class for
    calculating the adaptation. It is necessary so that the TemplateArray class
    can construct this linked list only once and then share it with all its
    Template attributes.
    '''
    cdef:
        scd.f_list **f_est_list;
        scd.fs_struct fs
        int k, p
        int num_chunks
        int chunk_len, sig_len_n

    def __cinit__(self, chunks, sig_len):
        self.sig_len_n = sig_len
        self.num_chunks = len(chunks)
        self.f_est_list = <scd.f_list**>malloc(self.sig_len_n*sizeof(scd.f_list*))
        for k in range(self.sig_len_n):
            self.f_est_list[k] = <scd.f_list*>malloc(sizeof(scd.f_list))
            scd.init_f_list(self.f_est_list[k])
        for k in range(self.num_chunks):
            chunk = chunks[k]
            chunk_len = len(chunk[0])
            for p in range(chunk_len):
                scd.fl_push(chunk[1][p], self.f_est_list[chunk[0][p]])

    def __dealloc__(self):
        for k in range(self.sig_len_n):
            # while self.f_est_list[k]->count > 0:
            #     scd.fl_pop(self.f_est_list[k])
            scd.free_f_list(self.f_est_list[k])
        free(self.f_est_list)

cdef class _finalizer:
    '''
    Simple extension type.  Used to assigned responsibility for freeing
    C-allocated heap memory to a python object. Taken from K.W. Smith's Cython
    book.
    '''
    cdef void *_data
    def __dealloc(self):
        if self._data is not NULL:
            free(self._data)



################################################################################
# Wrapped C Functions
################################################################################

def template_vals(double[:] freqs, double f0, double sigma, int num_h):
    cdef unsigned int length = len(freqs)
    return <double[:length]> scd.template_vals_c(&freqs[0], length, f0, sigma, num_h)

def wta_net(double[:,::1] E, double[:,::1] k, int num_n, int sig_len, double dt,
        double[:] tau, double M, double N, double sigma):
    cdef double *out = scd.wta_net_c(&E[0,0], &k[0,0], num_n, sig_len, dt, 
                                        &tau[0], M, N, sigma)
    cdef double [:,::1] out_mv = <double[:num_n,:sig_len]>out
    cdef np.ndarray out_np = np.asarray(out_mv)
    set_base(out_np, out)
    return out_np

cpdef tuple template_adapt(TemplateData td, double f0, int num_h, double sigma,
        double mu):
    cdef scd.fs_struct fs
    fs = scd.template_adapt_c(td.f_est_list, td.sig_len_n, f0, mu, num_h, sigma)
    cdef double[::1] freqs = <double[:td.sig_len_n]>fs.freqs;
    cdef double[::1] strengths = <double[:td.sig_len_n]>fs.strengths;
    return np.asarray(freqs), np.asarray(strengths)



################################################################################
# Cython Functions
################################################################################
cdef void set_base(np.ndarray array, void *carray):
    cdef _finalizer f = _finalizer()
    f._data = <void*>carray
    np.set_array_base(array, f)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple process_chunks(list chunks, int sig_len_n, double f0, double
        mu, int num_h, double sigma):
    '''
    recall that chunks is a list of tuples, and each tuple consists of two
    lists: one of indices, one of frequency values.
    '''
    # First reorganize the data into a series of linked lists for each time
    # index.
    cdef scd.f_list **f_est_list = <scd.f_list**>malloc(sig_len_n*sizeof(scd.f_list*))
    cdef scd.fs_struct fs
    cdef int k, p
    cdef int num_chunks = len(chunks)
    cdef int chunk_len
    for k in range(sig_len_n):
        f_est_list[k] = <scd.f_list*>malloc(sizeof(scd.f_list))
        scd.init_f_list(f_est_list[k])
    for k in range(num_chunks):
        chunk = chunks[k]
        chunk_len = len(chunk[0])
        for p in range(chunk_len):
            scd.fl_push(chunk[1][p], f_est_list[chunk[0][p]])

    # Then do the actual template adaptation
    fs = scd.template_adapt_c(f_est_list, sig_len_n, f0, mu, num_h, sigma)

    # Finally put the results into numpy arrays (via memory views) and return
    # the results.
    cdef double[::1] freqs = <double[:sig_len_n]>fs.freqs;
    cdef double[::1] strengths = <double[:sig_len_n]>fs.strengths;
    return np.asarray(freqs), np.asarray(strengths)

            

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple process_data(np.ndarray[np.float64_t] in_sig, double f_c, double bw, double f_s,
        np.ndarray[np.float64_t] b_lpf, np.ndarray[np.float64_t] a_lpf, double scale_fac,
        double min_e, double eps, double k_i, double k_p):
    '''
    Called FDL class.
    Function that actually performs the adaptive filtering. Generates a
    list of 3-element tuples (f0, times, output). All of these are needed
    to initiate the PLL stage.
    '''
    cdef int filt_ord = 2
    cdef int buf_size = filt_ord + 1
    # Allocate memory and circular buffer indices
    in_sig = np.concatenate((np.zeros(buf_size - 1), in_sig))
    cdef:
        int sig_len = len(in_sig)
        double dt = 1.0/f_s
        double f_c_base = f_c
        double env_diff
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
            # env_diff = ((env_u[idx0]*scale_fac)/env_c[idx0]) - \
            #             (env_l[idx0]/env_c[idx0]) 
            env_diff = log(env_u[idx0]) - log(env_l[idx0])
            if env_diff < eps: 
            # if env_diff == 0.0:
                if is_locked == 0:
                    is_locked = 1
                    on_record[num_on] = k - offset
                    num_on += 1

            else:
                if is_locked == 1:
                    is_locked = 0
                    off_record[num_on-1] = k - offset
                    ## TEST: reset after losing lock
                    # f_c = f_c_base
                    # out_l *= 0
                    # out_u *= 0
                    # env_l *= 0
                    # env_u *= 0
                    # u *= 0
        else:
            if is_locked == 1:
                is_locked = 0
                off_record[num_on-1] = k - offset
                ## TEST: reset after losing lock
                # f_c = f_c_base
                # out_l *= 0
                # out_u *= 0
                # env_l *= 0
                # env_u *= 0
                # u *= 0

        # Calculate the error, control equations, frequency update
        # scale factor inserted here to avoid messing with dynamics
        err[k] = log(scale_fac*env_u[idx0]) - log(env_l[idx0])    
        u[idx0] = u[idx1] + k_p*(err[k]-err[k-1]) + dt*k_i*err[k]
        
        ## original (basic) version -- unlimited range
        f_c = f_c_base + u[idx0]
        
        ## reset if outside of range
        if f_c > f_c_base + 0.75*bw:
            if is_locked == 1:
                is_locked = 0
                off_record[num_on-1] = k - offset
            f_c = f_c_base
            out_l *= 0
            out_u *= 0
            env_l *= 0
            env_u *= 0
            u *= 0
        if f_c < f_c_base - 0.75*bw:
            if is_locked == 1:
                is_locked = 0
                off_record[num_on-1] = k - offset
            f_c = f_c_base
            out_l *= 0
            out_u *= 0
            env_l *= 0
            env_u *= 0
            u *= 0
        
        f_record[k] = f_c

    return (out_c[offset:sig_len], f_record[offset:sig_len], on_record, off_record, num_on)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bp_narrow_coefs(double f_c, double bw, double f_s, 
                             np.ndarray[np.float64_t] b, 
                             np.ndarray[np.float64_t] a):
    '''
    From Steven W. Smith's book, The Scientist and Engineer's Guide to Digital 
    Signal Processing, eqs. 19-7 and 19-8. Note that his coefficient 
    conventions are different from Julius O. Smith's: a is b and b is a, and 
    the signs of the a coefficients (in JOS notation) are flipped. JOS notation
    is used here.
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
    Called by SCFB class.
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
        phi[k] = phi[k-1] - mu*lp_out[idx0]
        out[k] = cos(2*pi*f_c*t + phi[k])
    cdef np.ndarray freq_offset = np.gradient(phi[buf_size-1:sig_len], dt)/(2*pi)
    freq_offset = dsp.filtfilt(b_s, a_s, freq_offset)   #smooth freq estimates
    # return out[buf_size-1:], (freq_offset+f_c)
    return freq_offset+f_c


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray agc(np.ndarray[np.float64_t, ndim=1] in_sig, double mu, double ds):
    '''
    Called SCFB class.
    Implementation of a simple automatic gain control (AGC) unit. The approach
    follows the "naive" implementation given in Johnson and Sethares' book
    Telecommunication Breakdown.  mu sets the sensitivity/noise tradeoff. ds
    sets the desired power.
    '''
    cdef np.ndarray[np.float64_t] a, s
    cdef int sig_len = len(in_sig)
    cdef int k
    cdef double sign
    a = np.zeros(sig_len, dtype=np.float64)
    a[0] = 1.0
    s = np.zeros_like(a)
    for k in range(1,sig_len):
        s[k-1] = a[k-1]*in_sig[k-1]
        sign = 1.0 if a[k-1] > 0 else 0.0
        a[k] = a[k-1] - mu * sign * (s[k-1]*s[k-1]-ds)
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple template_calc(np.ndarray[np.float64_t, ndim=1] freqs, double f0, int num_h, double sigma):
    '''
    Called by template class (calculates the strengths and adjustments.
    '''
    cdef double dJ = 0.0
    cdef double temp = 0.0
    cdef double s = 0.0
    cdef double factor
    cdef int num_inputs = np.size(freqs)
    cdef int n, p
    for n in range(num_inputs):
        for p in range(1,num_h+1):
            # temp = exp( - ((freqs[n] - p*f0)**2)/(2 * (p * sigma)**2) )
            factor = 1.0 if p==1 else 0.8
            temp = exp( - ((freqs[n] - p*f0)**2)/(2 * (sigma)**2) )
            s += temp 
            dJ += ((freqs[n] - p*f0)/(p * sigma**2))*temp
    return dJ, s
    

