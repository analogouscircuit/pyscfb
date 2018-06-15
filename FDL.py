import numpy as np
import scipy.fftpack as fft
import scipy.signal as dsp
import matplotlib.pyplot as plt
import math
from simplefiltutils import bp_narrow_coefs

def peaking_coefs(f_c, bw, gain, f_s):
    c = 1.0/(np.tan(np.pi*f_c/f_s))
    cs = c**2
    csp1 = cs+1.0
    Bc = bw*c
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


def FDL(in_sig, f_c, bw_gt, gain, f_s):
    dt = 1.0/f_s

    # set up filter parameters and coefficients
    bw = bw_gt/4.0      #bw_gt is for whole channel -- gammatone ultimately, here butterworth
    f_l = f_c - bw
    f_u = f_c + bw
    # b_l, a_l = peaking_coefs(f_l, bw, gain, f_s)
    # b_c, a_c = peaking_coefs(f_c, bw, gain, f_s)
    # b_u, a_u = peaking_coefs(f_u, bw, gain, f_s)
    b_l, a_l = bp_narrow_coefs(f_l, bw, f_s)
    b_c, a_c = bp_narrow_coefs(f_c, bw, f_s)
    b_u, a_u = bp_narrow_coefs(f_u, bw, f_s)

    b_lpf, a_lpf = dsp.butter(2, (20.0*2)/f_s)
    
    # Calculate parameters for control loop
    tau_g = (dsp.group_delay( (b_c, a_c), [f_c*2/f_s] )[1] 
                + dsp.group_delay( (b_lpf, a_lpf), [f_c*2/f_s] )[1]) # group delay in samples
    tau_g = tau_g*dt    # and now in time
    tau_s = 50.0/f_c    # settling time
    _, H_u = dsp.freqz(b_u, a_u)   # transfer function squared for upper filter
    _, H_l = dsp.freqz(b_l, a_l)   # transfer function squared for lower filter
    H_us = np.real(H_u*np.conj(H_u))
    H_ls = np.real(H_l*np.conj(H_l))
    S = (H_us - H_ls)/(H_us + H_ls) # transfer function of frequency discriminator
    dS = np.gradient(S)     # derivative of S, needs to be evaluated at f_c
    f_step = (f_s/2.0)/len(dS)
    idx = math.floor(f_c/f_step)
    w = (f_c/f_step) - idx
    k_s = (1.0-w)*dS[idx] + w*dS[idx+1]     # frequency discriminator constant
    # k_i = 10.95 * tau_g / (k_s*(tau_s**2))  # PID integration term coefficient
    k_i = 0
    
    # Allocate memory and circular buffer indices
    in_sig = np.concatenate((np.zeros(2), in_sig))
    f_record = np.zeros_like(in_sig)
    err = np.zeros_like(in_sig)
    u   = np.zeros_like(in_sig)
    out_l = np.zeros(3, dtype=np.float32)
    out_c = np.zeros_like(in_sig)
    out_u = np.zeros(3, dtype=np.float32)
    env_l = np.zeros(3, dtype=np.float32)
    env_u = np.zeros(3, dtype=np.float32)
    idx0 = 2
    idx1 = 1
    idx2 = 0
    for k in range(2, len(in_sig)):
        idx0 = (idx0 + 1)%3
        idx1 = (idx1 + 1)%3
        idx2 = (idx2 + 1)%3
        t = (k-2)*dt
        
        # first run through the triplet of filters
        out_l[idx0] = b_l[0]*in_sig[k] + b_l[1]*in_sig[k-1] + b_l[2]*in_sig[k-2]\
                        - a_l[1]*out_l[idx1] - a_l[2]*out_l[idx2]
        out_c[k]    = b_c[0]*in_sig[k] + b_c[1]*in_sig[k-1] + b_c[2]*in_sig[k-2]\
                        - a_c[1]*out_c[k-1] - a_c[2]*out_c[k-2]
        out_u[idx0] = b_u[0]*in_sig[k] + b_u[1]*in_sig[k-1] + b_u[2]*in_sig[k-2]\
                        - a_u[1]*out_u[idx1] - a_u[2]*out_u[idx2]

        # now run outputs of outer filters through envelope detectors (rectify & LPF)
        env_l[idx0] = b_lpf[0]*np.abs(out_l[idx0]) + b_lpf[1]*np.abs(out_l[idx1]) + b_lpf[2]*np.abs(out_l[idx2])\
                        - a_lpf[1]*env_l[idx1] - a_lpf[2]*env_l[idx2]
        env_u[idx0] = b_lpf[0]*np.abs(out_u[idx0]) + b_lpf[1]*np.abs(out_u[idx1]) + b_lpf[2]*np.abs(out_u[idx2])\
                        - a_lpf[1]*env_u[idx1] - a_lpf[2]*env_u[idx2]

        # calculate the error, control equations, frequency update
        err[k] = np.log(env_u[idx0]) - np.log(env_l[idx0])
        u[k] = u[k-1] + dt*k_i*err[k]
        f_c = f_c + u[k]
        f_l = f_c - bw
        f_u = f_c + bw
        f_record[k] = f_c

        # update coefficients
        b_l, a_l = peaking_coefs(f_l, bw, gain, f_s)
        b_c, a_c = peaking_coefs(f_c, bw, gain, f_s)
        b_u, a_u = peaking_coefs(f_u, bw, gain, f_s)

    return out_c[2:], f_record[2:]

################################################################################
if __name__ == "__main__":
    fs = 44100
    dt = 1.0/fs
    dur = 1.0
    f_in = 910.0
    times = np.arange(0.0, dur, dt)
    in_sig = np.cos(2*np.pi*f_in*times)

    output, f_rec = FDL(in_sig, 900.0, 50.0, 2.0, fs)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(times,in_sig)
    ax1.plot(times,output)
    ax2.plot(f_rec)
    plt.show()
