import numpy as np
import scipy.signal as dsp
import scipy.fftpack as fft
import matplotlib.pyplot as plt


def filter_simp(b, a, sig):
    '''
    Straightfoward (i.e. naive) implementation of a causal filter with
    coefficients given in the standard form. Automatically sets initial
    conditions to 0.
    '''
    pad_len = max(len(b), len(a)) - 1
    sig = np.concatenate((np.zeros(pad_len, dtype=np.float32), sig))
    y   = np.zeros_like(sig)
    print(sig.shape)
    for k in range(pad_len, len(sig)):
        for j in range(len(b)):
            y[k] += b[j]*sig[k-j]
        for j in range(1,len(a)):
            y[k] -= a[j]*y[k-j]
    return y[pad_len:]
        

def calc_one_pole_coefs(f_3db, fs, gain=1.0):
   '''
   Calculates the coefficients for a simple one-pole LP-filter given the cutoff
   frequency (-3dB) and sample frequency.  Unity gain at DC is default.
   '''
   a = np.ones(2, dtype=np.float32)
   b = np.zeros_like(a)
   c = 8-2*np.cos(2*np.pi*f_3db/fs)
   a[1] = -(c - np.sqrt(c**2 - 36.0))/6.0
   b[0] = gain*np.sqrt(1.0 + 2.0*a[1] + a[1]**2)
   return b, a

def main():
   f_s = 44100
   T = 1./f_s
   dur = 1.0
   times = np.arange(0,dur,T)
   imp = np.zeros_like(times)
   imp[0] = 1.0
   b, a = calc_one_pole_coefs(5000.0, f_s)
   print("b:", b)
   print("a:", a)
   ir = filter_simp(b, a, imp)
   ir_spec = np.abs(fft.fft(ir))
   fig = plt.figure()
   ax1 = fig.add_subplot(2,1,1)
   ax2 = fig.add_subplot(2,1,2)
   ax1.plot(times, ir)
   ax2.plot(times, ir_spec)
   plt.show()
   

################################################################################
if __name__=="__main__":
    main()
