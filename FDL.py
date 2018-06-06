import numpy as np
import scipy.signal as dsp
import scipy.fftpack as fft
import matplotlib.pyplot as plt

def cos_cos_test(f_3db, f_trans, f_c, f_s, sig):
    '''
    b and a are the filter coefficient for the prototype filter.  f_c
    is the vco freqency (modulator of incoming signal, center frequency
    of the filtering structure); f_trans is the 
    frequency by which the outside filters are translated relative to
    the center frequency. f_s is the sampling frequency, sig the signal.
    '''
    times = np.arange(0,len(sig)/f_s, 1.0/f_s)
    vco_sin = np.sin(2*np.pi*f_c*times)
    vco_cos = np.cos(2*np.pi*f_c*times)
    sig_mod_sin = vco_sin * sig
    sig_mod_cos = vco_cos * sig
    b, a = cos_mod_filt_coefs(f_3db, f_trans, f_s)
    sig1 = filter_simp(b, a, sig_mod_cos) * vco_cos
    sig2 = filter_simp(b, a, sig_mod_sin) * vco_sin
    return sig1 + sig2
        

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
        

def cos_mod_filt_coefs(f_3db, f_mod, fs):
    '''
    Generates the coefficients for the cosine-modulated version of the prototype
    impulse response (h1 in the paper).
    '''
    b_1p, a_1p = calc_one_pole_coefs(f_3db, fs)
    b = np.zeros(3, dtype=np.float32)
    a = np.ones_like(b)
    b[0] = b_1p[0]
    b[1] = b_1p[0] * a_1p[1] * np.cos(2*np.pi*f_mod)
    a[1] = -2*a_1p[1]*np.cos(2*np.pi*f_mod)
    a[2] = a_1p[1]**2
    return b, a

def sin_mod_filt_coefs(f_3db, f_mod, fs):
    '''
    '''
    b_1p, a_1p = calc_one_pole_coefs(f_3db, fs)
    b = np.zeros(3, dtype=np.float32)
    a = np.ones_like(b)
    b[0] = 0
    b[1] = b_1p[0] * a_1p[1] * np.sin(2*np.pi*f_mod)
    a[1] = -2*a_1p[1]*np.cos(2*np.pi*f_mod)
    a[2] = a_1p[1]**2
    return b, a


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

def main():
   f_s = 44100
   T = 1./f_s
   dur = 1.0
   times = np.arange(0,dur,T)
   imp = np.zeros_like(times)
   imp[0] = 1.0
   # b, a = calc_one_pole_coefs(5000.0, f_s)
   b, a = sin_mod_filt_coefs(200.0, 50.0, f_s)
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
