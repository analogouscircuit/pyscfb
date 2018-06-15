import numpy as np
import scipy.signal as dsp
import scipy.fftpack as fft
import matplotlib.pyplot as plt

def bp_narrow_coefs(f_c, bw, f_s):
    '''
    From Steven W. Smith's book, The Scientist and Engineer's Guide to Digital 
    Signal Processing, eqs. 19-7 and 19-8. Note that his coefficient 
    conventions are different from Julius O. Smith's: a is b and b is a, and 
    the signs of the a coefficients (in JOS notation) are flipped.
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

def filter_simp(b, a, sig):
    '''
    Straightfoward (i.e. naive) implementation of a causal filter with
    coefficients given in the standard form. Automatically sets initial
    conditions to 0.
    '''
    pad_len = max(len(b), len(a)) - 1
    sig = np.concatenate((np.zeros(pad_len, dtype=np.float32), sig))
    y   = np.zeros_like(sig)
    for k in range(pad_len, len(sig)):
        for j in range(len(b)):
            y[k] += b[j]*sig[k-j]
        for j in range(1,len(a)):
            y[k] -= a[j]*y[k-j]
    return y[pad_len:]
        

def conv_simp(ir, sig):
    '''
    Straightforward (i.e. naive) implementation of convolution for FIR
    filtering
    '''
    out = np.zeros(len(sig) + len(ir) - 1, dtype=np.float32)
    for n in range(len(out)):
        for k in range(min(n, len(ir))):
            out[n] += out[n-k] * ir[k]
    return out

def calc_one_pole_coefs(f_3db, fs, gain=1.0):
   '''
   Calculates the coefficients for a simple one-pole LP-filter given the cutoff
   frequency (-3dB) and sample frequency.  Unity gain at DC is default.

   This is the "prototype" filter used for the FDL.  It corresponds to the
   continuous time impulse response of h(t) = exp(-alpha*abs(t)) (except
   causal).
   '''
   a = np.ones(2, dtype=np.float32)
   b = np.zeros_like(a)
   c = 8-2*np.cos(2*np.pi*f_3db/fs)
   a[1] = -(c - np.sqrt(c**2 - 36.0))/6.0
   b[0] = gain*np.sqrt(1.0 + 2.0*a[1] + a[1]**2)
   return b, a

def peaking_coefs(gain, f_c, bw, f_s):
    '''
    Calcuates the coefficients for a "peaking" filter. This is a 
    symmetrical filter with a peak at f_c, with bandwidth bw and
    specified gain.  
    '''
    c = np.tan(np.pi*(0.5 - f_c/f_s))
    # c = 1.0/np.tan(np.pi*f_c/f_s)
    cs = c**2
    csp1 = cs+1.0
    Bc = (bw/f_s)*c
    gBc = gain*Bc
    norm = 1.0/(csp1 + Bc)
    b = np.zeros(3,dtype=np.float32)
    a = np.ones_like(b)
    b[0] = (csp1 + gBc)*norm
    b[1] = 2.0*(1.0 - cs)*norm
    b[2] = (csp1 - gBc)*norm
    a[0] = 1.0
    a[1] = b[1]
    a[2] = (csp1 - Bc)*norm
    return b, a

################################################################################
if __name__ == "__main__":
    b, a = bp_narrow_coefs(900.0, 100.0, 44100.0)
    b, a = dsp.iirpeak(900.0/(44100/2), 10)
    print("Filter coefficients:\n b:", b,"a:", a)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    # impulse response by hand
    # fs = 44100
    # imp = np.zeros(fs, dtype=np.float32)
    # imp[0] = 1
    # ir = filter_simp(b, a, imp)
    # ax1.plot(np.abs(fft.fft(ir)))
    fs = 44100
    t = np.arange(0,0.5,1/fs)
    in_sig = np.cos(2*np.pi*900.0*t)
    out_sig = filter_simp(b, a, in_sig)
    ax1.plot(t, out_sig)
    # freqz analysis
    num_pts = 2048
    df = (fs/2)/num_pts
    w, H = dsp.freqz(b, a, 2048)
    w = (w/np.pi)*df
    ax2.plot(w, H)
    plt.show()
