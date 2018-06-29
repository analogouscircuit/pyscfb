import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import math
import pdb
import time
import pyximport
pyximport.install()
import scfbutils_c as scfb

class FDL:
    '''
    Implementation of the Frequency Discriminator Loop (FDL) as described in 
    Kumaresan, Peddinti and Cariani (2013). This is a central component of the
    Synchrony Capture Filter Bank (SCFB). The implementation differs from the
    paper, which give a presentation in terms of non-causal, continuous-time
    filters. We instead use FIR filters that have low-cost parameter
    calculations. These FIR filters are *not* perfectly symmetrical, so a
    somewhat ad-hoc adjustment term is added so that the triplet filter bank
    converges more nearly to the desired frequency. 
    '''
    def __init__(self, f_c, bw_gt, f_s, debug=False):
        self.f_s = f_s
        self.dt = 1.0/f_s
        self.f_c = f_c
        self.f_c_base = f_c
        self.debug=debug

        # set up filter parameters and coefficients
        self.bw = bw_gt/2.0      # bw_gt is for whole channel -- gammatone ultimately
                            # paper gives bw_gt/4.0 for some reason, 2.0 works better
        self.f_l = f_c - self.bw
        self.f_u = f_c + self.bw
        self.b_l, self.a_l = self.bp_narrow_coefs(self.f_l, self.bw, f_s)
        self.b_c, self.a_c = self.bp_narrow_coefs(self.f_c, self.bw, f_s)
        self.b_u, self.a_u = self.bp_narrow_coefs(self.f_u, self.bw, f_s)
        self.b_len = len(self.b_c)
        self.a_len = len(self.a_c)
        self.buf_size = max(self.a_len, self.b_len)    # for circular buffer allocation
        lp_cutoff = f_c/15.0     # 8.0-12.0 is a good range
        # lp_cutoff = 50.0
        # self.b_lpf, self.a_lpf = dsp.bessel(2, (lp_cutoff*2)/f_s)
        self.b_lpf, self.a_lpf = dsp.bessel(2, (lp_cutoff*2)/f_s)
        
        # Calculate parameters for control loop -- can be made much more efficient
        # by removing calculation of entire transfer function -- only use one point

        # First time/delay calculations
        tau_g = (dsp.group_delay( (self.b_c, self.a_c), np.pi*(f_c*2/f_s) )[1] 
                    + dsp.group_delay( (self.b_lpf, self.a_lpf), np.pi*(f_c*2/f_s) )[1]) 
        tau_g = tau_g*self.dt    # Convert from samples to time
        tau_s = 15.0/f_c    # Settling time -- paper gives 50.0/f_c, but
                            # 15.0 seems to work better with our filters

        # Calculate control loop parameters
        # Need to remove calculation of (many) unnecessary points
        num_points = 2**15
        freqs = np.arange(0, (f_s/2), (f_s/2)/num_points)
        _, H_u = dsp.freqz(self.b_u, self.a_u, num_points)   # upper filter TF
        _, H_l = dsp.freqz(self.b_l, self.a_l, num_points)   # lower filter TF
        H_us = np.real(H_u*np.conj(H_u))    # now squared
        H_ls = np.real(H_l*np.conj(H_l))    # now squared
        num = (H_us - H_ls)
        denom = (H_us + H_ls)
        S = np.zeros_like(H_us)
        idcs = denom != 0
        S[idcs] = num[idcs]/denom[idcs]
        f_step = (f_s/2.0)/num_points
        # dS = np.gradient(S, f_step)     # derivative of S, to be evaluated at f_c
        idx = math.floor(f_c/f_step)
        w = (f_c/f_step) - idx
        # k_s_grad = (1.0-w)*dS[idx] + w*dS[idx+1]     # freq. discriminator constant
        # k_i = 10.95 * tau_g / (k_s*(tau_s**2))  # paper implementation
        gamma = tau_g/2
        beta = 8.11*(gamma/tau_s) + 21.9*((gamma/tau_s)**2)

        self.k_s = (S[idx] - S[idx-1])/f_step # primiate gradient calc
                                              # other version commented out
                                              # above
        self.k_p = (1/self.k_s)*(beta-1)/(beta+1)
        self.k_i = (1.0/self.k_s)*(21.9*(gamma/(tau_s**2)))*(2.0/(beta+1))

        # Calculate compensatory factor for asymmetry of filters --
        # only works perfectly at center frequency, but still an
        # improvement.
        r_l = (1.0-w)*np.sqrt(H_ls[idx]) + w*np.sqrt(H_ls[idx+1])
        r_u = (1.0-w)*np.sqrt(H_us[idx]) + w*np.sqrt(H_us[idx+1])
        self.scale_fac = r_l/r_u
        # self.scale_fac = 1.       # to compare without correction

        self.eps = 1e-24 # threshold for determining locked condition
        self.min_e = 0.01    # minimum energy for locking condition

    def process_data(self, in_sig):
        '''
        Version of below in Cython (about 150 times faster)
        '''
        out, self.f_record, on, off, num_on = scfb.process_data(
                          in_sig, self.f_c, self.bw, self.f_s, self.b_lpf,
                          self.a_lpf, self.scale_fac, self.min_e, self.eps,
                          self.k_i, self.k_p)
        self.out = out
        self.idx_chunks = []
        self.out_chunks = []
        self.freq_chunks = []
        for k in range(num_on):
            if off[k] == 0:
                off[k] = len(in_sig)-1
            self.idx_chunks.append(np.arange(on[k], off[k]+1))
            self.out_chunks.append(out[on[k]:off[k]+1])
            self.freq_chunks.append(self.f_record[on[k]:off[k]+1])
        return [f[0] for f in self.freq_chunks], self.idx_chunks, self.out_chunks, num_on


    @staticmethod
    def bp_narrow_coefs(f_c, bw, f_s):
        '''
        From Steven W. Smith's book, The Scientist and Engineer's Guide to Digital 
        Signal Processing, eqs. 19-7 and 19-8. Note that his coefficient 
        conventions are different from Julius O. Smith's: a is b and b is a, and 
        the signs of the a coefficients (in JOS notation) are flipped.
        '''
        f  = f_c/f_s
        bw = bw/f_s
        #pdb.set_trace()
        R = 1.0 - 3*bw
        K = (1.0 - 2*R*np.cos(2*np.pi*f) + (R**2))/(2 - 2*np.cos(2*np.pi*f))
        b = np.zeros(3,dtype=np.float32)
        a = np.ones_like(b)
        b[0] = 1.0 - K
        b[1] = 2*(K-R)*np.cos(2*np.pi*f)
        b[2] = (R**2)-K
        a[1] = -2*R*np.cos(2*np.pi*f)
        a[2] = (R**2)
        return b, a



def pll(in_sig, f_c, f_s):
    '''
    Implementation of a digital phase-locked loop (PLL) as presented in
    Sethares' "Rhythm and Transforms" book (2007).  Uses a gradient-ascent
    approach, correlating the input signal with an internal oscillator,
    adjusting the phase of the internal oscillator to maximize the correlation.
    The frequency estimate is found by calculating the gradient of the phase.

    Note that this is slow! There is a faster (>100x) cpdef version in
    scfbutils_c.pyx.
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

################################################################################
if __name__ == "__main__":
    f_s = 44100
    dt = 1.0/f_s
    dur = 0.5
    f_in_0 = 1200.0
    f_in_1 = 1200.0
    times = np.arange(0.0, dur, dt)
    f = np.linspace(f_in_1, f_in_0, len(times))
    in_sig = np.cos(2*np.pi*f*times)
    fdl = FDL(f_in_0*0.99, 250.0, f_s)
    num_its = 1 

    start = time.time()
    for k in range(num_its):
        f0s, idcs, outs = fdl.process_data_py(in_sig)
    end = time.time()
    print("Python implementation: %.4f (s) per call" % ( (end-start)/num_its ) )
    plt.plot(fdl.out)

    start = time.time()
    for k in range(num_its):
        f0s, idcs, outs = fdl.process_data(in_sig)
    end = time.time()
    print("Cython implementation: %.4f (s) per call" % ( (end-start)/num_its ) )
    plt.plot(fdl.out)
    # plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    for k in range(len(f0s)):
        ax1.plot(idcs[k], fdl.freq_chunks[k])
    ax2.plot(np.arange(0,len(in_sig)),fdl.f_record)
    ax2.plot(np.arange(0,len(in_sig)), f)
    plt.show()
