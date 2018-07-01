import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import scipy.io.wavfile
import pdb
import pyximport
import pickle
from FDL import FDL
from Template import Template
pyximport.install(setup_args={"include_dirs":np.get_include()})
from scfbutils_c import pll, agc


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
            # b, a = dsp.bessel(4, np.array([max(self.f_c[k] - self.bw[k],
            #     self.f_c[0]), self.f_c[k] + self.bw[k]])*(2/f_s),
            #     btype='bandpass')
            # b, a = dsp.bessel(2, np.array([max(self.f_c[k] - 0.5*self.bw[k],
            #     self.f_c[0]), self.f_c[k] + 0.5*self.bw[k]])*(2/f_s),
            #     btype='bandpass')
            b, a = dsp.bessel(2, np.array([max(self.f_c[k] - 0.5*self.bw[k],
                20.0), self.f_c[k] + 0.5*self.bw[k]])*(2/f_s),
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
                out_chunks[j] = agc(out_chunks[j], 0.1, 0.25)
                out_chunks[j] = agc(out_chunks[j], 0.001, 0.25)
                # agc_out_chunks.append(out_chunks[j])
                freq_est = pll(out_chunks[j], f0s[j], self.f_s)
                self.chunks.append( (idx_chunks[j], freq_est) )
        self.processed = True
        return self.chunks    # final goal, next one is for debuggin
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
        print("num chunks: ", len(self.chunks))
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
def stepped_test_sig(dur, num_seg, f0, f1, fs):
    num_samps = dur*fs
    seg_len = int(num_samps//num_seg)
    sig = np.zeros(seg_len*num_seg)
    f_step = (f1-f0)/num_seg
    f = np.arange(f0, f1, f_step)
    for k in range(num_seg):
        sig[k*seg_len:(k+1)*seg_len] = np.cos(2*np.pi*f[k]*np.arange(0,seg_len)/fs)
    return sig
    

if __name__=="__main__":
    # scfb = SCFB(559.56, 559.56, 1, 44100)
    scfb = SCFB(200., 4000., 100, 44100)
    f_s = 44100
    dt = 1./f_s
    f_0 = 1800.0
    f_1 = 1.1*f_0
    num_h = 2
    dur = 0.5
    t = np.arange(0, dur, dt)
    f_vals = np.linspace(f_0, f_1, len(t))
    in_sig1 = np.zeros_like(t)
    in_sig2 = np.zeros_like(t)
    for k in range(1,num_h+1):
        in_sig1 += (1./k)*np.cos(2.*np.pi*k*f_0*t)
        in_sig2 += (1./k)*np.cos(2.*np.pi*k*f_1*t)
    in_sig = in_sig1
    # in_sig = np.concatenate((in_sig1, in_sig2))

    ## use a stepped sequence
    in_sig = stepped_test_sig(1.0, 5, 500, 1000, f_s)
    
    ## read a signal from a file
    # f_s, in_sig = scipy.io.wavfile.read("/home/dahlbom/audio/audio_files/beethoven_1s.wav")
    # in_sig = np.array(in_sig, dtype=np.float32)
    # in_sig = in_sig/(2**15)
    # in_sig = in_sig/np.max(in_sig)
    # print("Max value of signal: ", np.max(in_sig))

    ## add noise to the signal
    noise = np.random.normal(0.0, 0.0001, size=len(in_sig))
    in_sig += noise

    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    # fdl_out, agc_out, idx_chunks = scfb.process_signal(in_sig, verbose=True)
    scfb.process_signal(in_sig, verbose=True)
    for fdl in scfb.fdl:
        ax.plot(np.arange(0,len(in_sig))/f_s, fdl.f_record, linewidth=0.75, color='k', linestyle='--')
    scfb.plot_output(ax)

    # fig2 = plt.figure()
    # ax1 = fig2.add_subplot(1,2,1)
    # ax2 = fig2.add_subplot(1,2,2)
    # for k in range(len(fdl_out)):
    #     ax1.plot(fdl_out[k]+k)
    #     ax2.plot(agc_out[k]+k)

    # for k in range(1,num_h+1):
    #     ax.plot(np.arange(0,len(in_sig))/f_s, k*f_vals, color='b')
    # ordered_out = scfb.get_ordered_output()
    # print(ordered_out[:100])

    ## spectrogram
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(1,1,1)
    # f, t, specgram = dsp.spectrogram(in_sig, f_s, nperseg=1024)
    # ax2.pcolormesh(t, f, specgram)
    # ax2.set_ylabel('Frequency (Hz)')
    # ax2.set_xlabel('Time (s)')

    plt.show()


    pickle.dump(scfb.get_ordered_output(), open("ordered_output.pkl", "wb"))

    # temp = Template(550.0, 5)
    # pitch, strength = temp.process_input(scfb.get_ordered_output())



