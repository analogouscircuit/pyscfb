import numpy as np
import scipy.fftpack as fft
import scipy.signal as dsp
import matplotlib.pyplot as plt
import math
from simplefiltutils import bp_narrow_coefs


def filt_chunk(in_chunk, out_chunk, b, a):
    '''
    Calculates one sample (to render main loop below more readable)
    Ultimately in-lined all this -- slated for removal
    '''
    b_len = len(b)
    a_len = len(a)
    return np.sum(in_chunk[-b_len:]*np.flip(b,0)) - np.sum(out_chunk[-a_len:-1]*np.flip(a[1:],0))



def FDL(in_sig, f_c, bw_gt, f_s):
    dt = 1.0/f_s
    f_c_base = f_c

    # set up filter parameters and coefficients
    bw = bw_gt/2.0      # bw_gt is for whole channel -- gammatone ultimately, here butterworth
                        # paper gives bw_gt/4.0 for some reason, 2.0 works better
    f_l = f_c - bw
    f_u = f_c + bw
    b_l, a_l = bp_narrow_coefs(f_l, bw, f_s)
    b_c, a_c = bp_narrow_coefs(f_c, bw, f_s)
    b_u, a_u = bp_narrow_coefs(f_u, bw, f_s)
    b_len = len(b_c)
    a_len = len(a_c)
    buf_size = max(a_len, b_len)    # for circular buffer allocation
    
    lp_cutoff = f_c/8.0     # this is essentially a free parameter -- 8.0-12.0 is a good range
    b_lpf, a_lpf = dsp.bessel(2, (lp_cutoff*2)/f_s)
    
    # Calculate parameters for control loop -- can be made much more efficient
    # by removing calculation of entire transfer function -- only use one point

    # First time/delay calculations
    tau_g = (dsp.group_delay( (b_c, a_c), np.pi*(f_c*2/f_s) )[1] 
                + dsp.group_delay( (b_lpf, a_lpf), np.pi*(f_c*2/f_s) )[1]) # group delay in samples
    tau_g = tau_g*dt    # and now in time
    tau_s = 15.0/f_c    # settling time -- paper gives 50.0/f_c

    # now calculate control loop constants
    num_points = 2**15
    freqs = np.arange(0, (f_s/2), (f_s/2)/num_points)
    _, H_u = dsp.freqz(b_u, a_u, num_points)   # transfer function for upper filter
    _, H_l = dsp.freqz(b_l, a_l, num_points)   # transfer function for lower filter
    H_us = np.real(H_u*np.conj(H_u))    # now squared
    H_ls = np.real(H_l*np.conj(H_l))    # now squared
    num = (H_us - H_ls)
    denom = (H_us + H_ls)
    S = np.zeros_like(H_us)
    idcs = denom != 0
    S[idcs] = num[idcs]/denom[idcs]
    f_step = (f_s/2.0)/num_points
    dS = np.gradient(S, f_step)     # derivative of S, needs to be evaluated at f_c
    idx = math.floor(f_c/f_step)
    w = (f_c/f_step) - idx
    k_s_grad = (1.0-w)*dS[idx] + w*dS[idx+1]     # frequency discriminator constant
    k_s = (S[idx] - S[idx-1])/f_step
    k_i = 10.95 * tau_g / (k_s*(tau_s**2))  # PID integration term coefficient -- paper implementation
    gamma = tau_g/2
    beta = 8.11*(gamma/tau_s) + 21.9*((gamma/tau_s)**2)
    k_p = (1/k_s)*(beta-1)/(beta+1)
    k_i = (1.0/k_s)*(21.9*(gamma/(tau_s**2)))*(2.0/(beta+1))

    # now calculate compensatory factor for asymmetry of filters
    r_l = (1.0-w)*np.sqrt(H_ls[idx]) + w*np.sqrt(H_ls[idx+1])
    r_u = (1.0-w)*np.sqrt(H_us[idx]) + w*np.sqrt(H_us[idx+1])
    scale_fac = r_l/r_u
    print(r_l)
    print(r_u)
    print(scale_fac)
    
    # Allocate memory and circular buffer indices
    in_sig = np.concatenate((np.zeros(buf_size -1), in_sig))
    f_record = np.zeros_like(in_sig)
    err = np.zeros_like(in_sig)
    u   = np.zeros_like(in_sig)
    out_l = np.zeros(buf_size, dtype=np.float32)
    out_c = np.zeros_like(in_sig)
    out_u = np.zeros(buf_size, dtype=np.float32)
    env_l = np.zeros(buf_size, dtype=np.float32)
    env_u = np.zeros(buf_size, dtype=np.float32)
    env_l_rec = np.zeros_like(in_sig)
    env_u_rec = np.zeros_like(in_sig)
    out_l_rec = np.zeros_like(in_sig)
    out_u_rec = np.zeros_like(in_sig)
    idx0 = 2
    idx1 = 1
    idx2 = 0
    err_integral = 0.0
    for k in range(buf_size-1, len(in_sig)):
        idx0 = (idx0 + 1)%buf_size
        idx1 = (idx1 + 1)%buf_size
        idx2 = (idx2 + 1)%buf_size
        
        out_l[idx0] = np.sum(in_sig[k-b_len+1:k+1]*np.flip(b_l,0)) - np.sum(np.roll(out_l,-idx0-1)[:-1]*np.flip(a_l[1:],0))
        out_c[k]    = np.sum(in_sig[k-b_len+1:k+1]*np.flip(b_c,0)) - np.sum(out_c[k-a_len+1:k]*np.flip(a_c[1:],0))
        out_u[idx0] = np.sum(in_sig[k-b_len+1:k+1]*np.flip(b_u,0)) - np.sum(np.roll(out_u,-idx0-1)[:-1]*np.flip(a_u[1:],0))

        out_l_rec[k] = out_l[idx0]
        out_u_rec[k] = out_u[idx0]

        # now run outputs of outer filters through envelope detectors (rectify & LPF)
        env_l[idx0] = b_lpf[0]*np.abs(out_l[idx0]) + b_lpf[1]*np.abs(out_l[idx1]) + b_lpf[2]*np.abs(out_l[idx2])\
                        - a_lpf[1]*env_l[idx1] - a_lpf[2]*env_l[idx2]
        env_u[idx0] = b_lpf[0]*np.abs(out_u[idx0]) + b_lpf[1]*np.abs(out_u[idx1]) + b_lpf[2]*np.abs(out_u[idx2])\
                        - a_lpf[1]*env_u[idx1] - a_lpf[2]*env_u[idx2]
        env_l_rec[k] = env_l[idx0]
        env_u_rec[k] = env_u[idx0]

        # calculate the error, control equations, frequency update
        err[k] = np.log(scale_fac*env_u[idx0]) - np.log(env_l[idx0])
        err_integral += err[k]*dt
        u[k] = u[k-1] + k_p*err[k] + dt*k_i*err[k] - k_p*err[k-1]

        f_c = f_c_base + u[k]
        f_l = f_c - bw
        f_u = f_c + bw
        f_record[k] = f_c

        # update coefficients
        b_l, a_l = bp_narrow_coefs(f_l, bw, f_s)
        b_c, a_c = bp_narrow_coefs(f_c, bw, f_s)
        b_u, a_u = bp_narrow_coefs(f_u, bw, f_s)


    # plot the history the FDL states -- only for debugging, will eventually remove
    # fig = plt.figure()
    # times = np.arange(0,(len(in_sig)-2))*dt
    # ax1 = fig.add_subplot(2,3,1)
    # ax2 = fig.add_subplot(2,3,2)
    # ax3 = fig.add_subplot(2,3,3)
    # ax4 = fig.add_subplot(2,3,4)
    # ax5 = fig.add_subplot(2,3,5)
    # ax6 = fig.add_subplot(2,3,6)
    # ax1.plot(times, in_sig[2:])
    # ax1.plot(times, out_c[2:])
    # ax2.plot(times, out_l_rec[2:])
    # ax3.plot(times, out_u_rec[2:])
    # ax4.plot(times, err[2:])
    # ax4.plot(times, u[2:])
    # ax5.plot(times, env_l_rec[2:])
    # ax6.plot(times, env_u_rec[2:])
    # fig2 = plt.figure()
    # ax1_1 = fig2.add_subplot(1,1,1)
    # ax1_1.plot(times, f_record[2:])
    # plt.show()
    return out_c[buf_size-1:], f_record[buf_size-1:]

################################################################################
if __name__ == "__main__":
    f_s = 44100
    dt = 1.0/f_s
    dur = 0.5
    f_in = 250.0
    times = np.arange(0.0, dur, dt)
    in_sig = np.cos(2*np.pi*f_in*times)

    f_c = f_in*0.95 
    bw = f_c*(2**(1/6)) - f_c/(2**(1/6))
    print("f_c: ", f_c)
    print("bw:  ", bw)
    output, f_rec = FDL(in_sig, f_c, bw, f_s)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.plot(times, in_sig)
    ax1.plot(times, output)
    ax2.plot(times, f_rec)
    plt.show()

