import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scfb
import pickle

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
    peri = scfb.SCFB(200., 4000., 100, 44100)
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
    # fdl_out, agc_out, idx_chunks = peri.process_signal(in_sig, verbose=True)
    peri.process_signal(in_sig, verbose=True)
    for fdl in peri.fdl:
        ax.plot(np.arange(0,len(in_sig))/f_s, fdl.f_record, linewidth=0.75, color='k', linestyle='--')
    peri.plot_output(ax)

    # fig2 = plt.figure()
    # ax1 = fig2.add_subplot(1,2,1)
    # ax2 = fig2.add_subplot(1,2,2)
    # for k in range(len(fdl_out)):
    #     ax1.plot(fdl_out[k]+k)
    #     ax2.plot(agc_out[k]+k)

    # for k in range(1,num_h+1):
    #     ax.plot(np.arange(0,len(in_sig))/f_s, k*f_vals, color='b')
    # ordered_out = peri.get_ordered_output()
    # print(ordered_out[:100])

    ## spectrogram
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(1,1,1)
    # f, t, specgram = dsp.spectrogram(in_sig, f_s, nperseg=1024)
    # ax2.pcolormesh(t, f, specgram)
    # ax2.set_ylabel('Frequency (Hz)')
    # ax2.set_xlabel('Time (s)')

    plt.show()


    pickle.dump(peri.get_ordered_output(), open("ordered_output.pkl", "wb"))

    # temp = Template(550.0, 5)
    # pitch, strength = temp.process_input(peri.get_ordered_output())



