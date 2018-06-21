import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
from scfbutils import FDL, pll

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
    def __init__(self, f_lo, f_hi, num_chan, f_s, verbose=False):
        # basic parameters and placeholders
        self.f_s = f_s
        self.dt = 1./f_s
        self.num_chan = num_chan
        self.chunks = []

        # calculate frequencies and bandwidths of channels
        self.f_c = np.logspace(np.log10(f_lo), np.log10(f_hi), num_chan)
        c = 2.**(1./6.) - 1/(2.**(1./6.))   # bw multiplier
        self.bw = [ max(100.0, f_c*c) for f_c in self.f_c ] 

        # Set up filter coefficients for each channel
        self.a = []
        self.b = []
        for k in range(self.num_chan):
            b, a = dsp.butter(2, np.array([max(self.f_c[k] - self.bw[k],
                self.f_c[0]), self.f_c[k] + self.bw[k]])*(2/f_s),
                btype='bandpass')
            self.a.append(a)
            self.b.append(b)

        # Set up FDLs and PLLs for each channel
        self.fdl = [FDL(self.f_c[k], self.bw[k], self.f_s) for k in
                        range(self.num_chan)]
        # self.pll = [PLL(self.f_c[k], self.f_s) for k in range(self.num_chan)]


    def process_signal(self, in_sig):
        for k in range(self.num_chan):
            filted = dsp.filtfilt(self.b[k], self.a[k], in_sig)
            f0s, idx_chunks, out_chunks = self.fdl[k].process_data(filted)





    def plot_output(self, in_sig):
        pass


    def get_ordered_output(self, in_sig):
        pass


################################################################################
if __name__=="__main__":
    scfb = SCFB(80, 4000, 50, 44100, verbose=True)
