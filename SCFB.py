import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import scipy.io.wavfile
import pdb
from FDL import FDL
from scfbutils_c import pll




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
            b, a = dsp.butter(4, np.array([max(self.f_c[k] - self.bw[k],
                self.f_c[0]), self.f_c[k] + self.bw[k]])*(2/f_s),
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
        for k in range(self.num_chan):
            if verbose:
                print("Processing channel %d/%d"%(k+1, self.num_chan))
            filted = dsp.filtfilt(self.b[k], self.a[k], in_sig)
            f0s, idx_chunks, out_chunks, num_chunks = self.fdl[k].process_data(filted)
            for j in range(num_chunks):
                if len(out_chunks[j]) < np.floor(0.01/self.dt):   # dur > 10 ms
                    continue
                freq_est = pll(out_chunks[j], f0s[j], self.f_s)
                self.chunks.append( (idx_chunks[j], freq_est) )
        self.processed = True
        return self.chunks


    def plot_output(self):
        '''
        Makes a simple plot of all captured frequency information
        '''
        if self.processed == False:
            print("You haven't processed an input yet!")
            return
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        print("num chunks: ", len(self.chunks))
        for k in range(len(self.chunks)):
            ax1.plot(self.chunks[k][0]*self.dt, self.chunks[k][1], color='k')
        plt.show()


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
        return self.ordered


################################################################################
if __name__=="__main__":
    scfb = SCFB(200, 2000, 10, 44100)
    f_s = 44100
    dt = 1./f_s
    f_0 = 220.0
    dur = 0.5
    t = np.arange(0, dur, dt)
    for k in range(1,2):
        in_sig = np.cos(2.*np.pi*k*f_0*t)
    # f_s, in_sig = scipy.io.wavfile.read("/home/dahlbom/audio/audio_files/beethoven_1s.wav")
    # in_sig = np.array(in_sig, dtype=np.float32)
    # in_sig = in_sig/(2**15)
    # in_sig = in_sig/np.max(in_sig)
    # print("Max value of signal: ", np.max(in_sig))
    scfb.process_signal(in_sig, verbose=True)
    scfb.plot_output()
    # ordered_out = scfb.get_ordered_output()
    # print(ordered_out[:100])

