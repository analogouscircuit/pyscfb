import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np
import time

################################################################################
# Functions
################################################################################
def yin_block(data, fs, w_size=None, tau_max=None, thresh = 0.5):
    '''
    Returns a pitch estimate for a windowed chunk of signal
    '''
    if w_size == None:
        w_size = len(data)//2
    if tau_max == None:
        tau_max = int(w_size//2)
    assert tau_max < w_size
    r = np.zeros(tau_max)
    ## difference function (autocorrelation-like)
    for i in range(tau_max):
        vec = data[:w_size] - data[i:w_size+i]
        r[i] = np.dot(vec, vec)
    t1 = time.time()
    ## d calculation
    d = np.zeros(tau_max)
    s = r[0]
    d[0] = 1
    for i in range(1, tau_max):
        s += r[i]
        d[i] = r[i] / ((1/i)*s)
    ## pick minimum (first min below threshold)
    idcs = np.where(d < thresh) 
    if len(d[idcs]) == 0:
        return 0
    
    return fs/idcs[0][np.argmin(d[idcs])]

def yin_series(data, fs, w_size, tau_max, hop_size, thresh=0.5):
    assert hop_size <= w_size
    num_hops = (len(data) - w_size - tau_max)//(hop_size) + 1
    p = np.zeros(num_hops) 
    idx0 = 0
    # for n in range(num_hops):
    idx1 = idx0 + w_size + tau_max
    sig_len = len(data)
    # while idx1 < sig_len:
    for n in range(num_hops):
        p[n] = yin_block(data[idx0:idx1], fs, w_size, tau_max,
                thresh)
        idx0 += hop_size
        idx1 += hop_size
    time_hop = hop_size/fs
    t = np.linspace(0, time_hop*num_hops, num_hops)
    t += time_hop/2
    return p, t 


################################################################################
# Main
################################################################################
if __name__=="__main__":
    fs, data = wavfile.read("E1_bass.wav")
    print("Length of sample: ", len(data)/fs)
    t0 = time.time()
    f0_lo = 30.0
    w_size = int((fs/f0_lo))
    print("window length: {} samples, {} seconds".format(w_size, w_size/fs))
    tau_max = w_size -1  
    hop_size = int(w_size//4)
    p_series, t = yin_series(data, fs, w_size, tau_max, hop_size)
    t1 = time.time()
    print("Time for pitch calculation with hop_size {}:".format(hop_size), t1-t0)
    plt.plot(t, p_series)
    plt.show()
