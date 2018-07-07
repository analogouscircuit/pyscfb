'''
The main Synchrony Capture Filter Bank (SCFB) module file.  This is pure python.
All C functionality is invoked through importation of the scfbutils (Cython)
module.  

The main classes defined here are the Frequency Discriminator Loop (FDL), which
is used in the construction of the entire SCFB class (which maintains an array
of FDLs as well as BM filters to preceded them).  
'''


import math
import numpy as np
import scipy.signal as dsp
import pdb
import scfbutils    # this is the Cython module


################################################################################
class SCFB:
    '''
    Implementation of Kumaresan, Peddinti and Cariani's (2013)
    Synchrony-Capture Filter Bank (SCFB). Consists of an array of 1/3rd-octave
    width bandpass filters (just Butterworth for now -- later we may get more
    sophisticated) simulating the Basilar Membrane.  Each channel is fed into a
    Frequency Discriminator Loop (FDL), implemented in a separated class.  The
    output of the FDLs are fed into phase-locked loops (PLLs -- also
    implemented in a separated class), when the FDLs report a locked
    condition.
    '''
    def __init__(self, f_lo, f_hi, num_chan, f_s):
        # basic parameters and placeholders
        self.f_s = f_s
        self.dt = 1./f_s
        self.num_chan = num_chan
        self.chunks = []
        self.processed = False

        # calculate frequencies and bandwidths of channels
        self.f_c = np.logspace(np.log10(f_lo), np.log10(f_hi), num_chan)
        c = 2.**(1./6.) - 1/(2.**(1./6.))   # bw multiplier
        self.bw = [ max(100.0, f_c*c) for f_c in self.f_c ] 
        print(self.f_c)

        # Set up filter coefficients for each channel
        self.a = []
        self.b = []
        for k in range(self.num_chan):
            b, a = dsp.bessel(2, np.array([max(self.f_c[k] - 0.5*self.bw[k],
                15.0), self.f_c[k] + 0.5*self.bw[k]])*(2/f_s),
                btype='bandpass')
            self.a.append(a)
            self.b.append(b)

        # Set up FDLs for each channel
        self.fdl = [FDL(self.f_c[k], self.bw[k], self.f_s) for k in
                        range(self.num_chan)]


    def process_signal(self, in_sig, verbose=False):
        '''
        Where all the actual signal processing is done -- actually, where all
        the signal processing functions and methods are called. Results are all
        stored in "self.chunks," which is a collection of ti
        '''
        self.in_sig = in_sig
        # fdl_out_chunks = []
        # agc_out_chunks = []
        for k in range(self.num_chan):
            if verbose:
                print("Processing channel %d/%d"%(k+1, self.num_chan))
            filted = dsp.filtfilt(self.b[k], self.a[k], in_sig)
            f0s, idx_chunks, out_chunks, num_chunks = self.fdl[k].process_data(filted)
            for j in range(num_chunks):
                if len(out_chunks[j]) < np.floor(0.03/self.dt):   # dur > 30 ms
                    continue
                # fdl_out_chunks.append(out_chunks[j])
                out_chunks[j] = scfbutils.agc(out_chunks[j], 0.1, 0.25)
                out_chunks[j] = scfbutils.agc(out_chunks[j], 0.001, 0.25)
                # agc_out_chunks.append(out_chunks[j])
                freq_est = scfbutils.pll(out_chunks[j], f0s[j], self.f_s)
                assert len(freq_est)==len(idx_chunks[j])
                self.chunks.append( (idx_chunks[j], freq_est) )
        self.processed = True
        return self.chunks    # final goal, next one is for debugging
        # return fdl_out_chunks, agc_out_chunks, idx_chunks


    def plot_output(self, ax=None):
        '''
        Makes a simple plot of all captured frequency information
        '''
        if self.processed == False:
            print("You haven't processed an input yet!")
            return
        if ax==None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        # print("num chunks: ", len(self.chunks))
        for k in range(len(self.chunks)):
            ax.plot(self.chunks[k][0]*self.dt, self.chunks[k][1], color='r', linewidth=0.5)
        # plt.show()


    def get_ordered_output(self):
        '''
        Returns a single list with the same number of elements as there were
        samples in the input signal.  Each element of the list is itself a
        list, to which is appended all frequencies that are detected at that
        time.  (The number of detected frequencies at each time is variable.)
        '''
        if self.processed == False:
            print("You haven't processed an input yet!")
            return
        self.ordered = [ [] for k in range(len(self.in_sig)) ]
        for n in range(len(self.chunks)):
            k = 0
            for idx in self.chunks[n][0]:
                self.ordered[idx].append(self.chunks[n][1][k])
                k += 1
        for k in range(len(self.ordered)):
            self.ordered[k] = np.array(self.ordered[k], dtype=np.float64)
        return self.ordered



################################################################################
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

        self.eps = 1e-32 # threshold for determining locked condition
        self.min_e = 0.1    # minimum energy for locking condition
        # self.min_e = 100./self.f_c   # allow less energy in upper reginos (1/f)

    def process_data(self, in_sig):
        '''
        Version of below in Cython (about 150 times faster)
        '''
        out, self.f_record, on, off, num_on = scfbutils.process_data(
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
        the signs of the a coefficients (in JOS notation) are flipped. JOS
        notational conventions are used here.

        Note that there is a Cython version of this implemented for the main
        loop (though declared as cdef, so inaccessible here).  This is only used
        in the initialization of an instance of the FDL class.
        '''
        f  = f_c/f_s
        bw = bw/f_s
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


################################################################################
class Template:
    '''
    '''
    def __init__(self, f0, num_h, sigma=10.0, mu=0.1):
        self.f0 = f0
        self.num_h = num_h
        self.mu = mu
        self.sig = sigma
        self.f_vals = []
        self.strengths = []

    def process_input(self, ordered_input):
        '''
        Main loop -- note that only the math is done in C. This is because of
        the algorithm's reliance on Python's dynamic data structures. This will
        be modified... 
        '''
        sig_len = len(ordered_input)
        phi = np.zeros(sig_len)
        s = np.zeros_like(phi)
        phi[0] = self.f0
        for k in range(sig_len-1):
            dJ, s[k] = scfbutils.template_calc(ordered_input[k], phi[k], 5, 10.0)
            phi[k+1] = phi[k] + self.mu*dJ
        self.f_vals = phi
        self.strengths = s
        return phi, s

    def process_chunks(self, chunks, sig_len_n):
        '''
        Full C implementation.
        '''
        phi, s = scfbutils.process_chunks(chunks, sig_len_n, self.f0, self.mu, self.num_h, self.sig)
        self.f_vals = phi
        self.strengths = s
        return phi, s

    def adapt(self, td):
        phi, s = scfbutils.template_adapt(td, self.f0, self.num_h, self.sig,
                self.mu)
        self.f_vals = phi
        self.strengths = s


class TemplateArray:
    def __init__(self, chunks, sig_len, f0_vals, num_h, sigma, mu):
        self.data = scfbutils.TemplateData(chunks, sig_len)
        self.templates = []
        for f0 in f0_vals:
            self.templates.append(Template(f0, num_h, sigma, mu))
            print(self.templates[-1].f0)

    def adapt(self):
        for k, t in enumerate(self.templates):
            # print("Adapting template {}".format(k+1))
            t.adapt(self.data)

