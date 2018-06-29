import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import pickle
import pyximport
pyximport.install()
import scfbutils_c as scfb


class Template:
    '''
    '''
    def __init__(self, f0, num_h, mu=0.1):
        self.f0 = f0
        self.num_h = num_h
        self.mu = mu
        self.sig = 50

    def process_input(self, ordered_input):
        sig_len = len(ordered_input)
        phi = np.zeros(sig_len)
        s = np.zeros_like(phi)
        phi[0] = self.f0
        for k in range(sig_len-1):
            dJ, s[k] = scfb.template_calc(ordered_input[k], phi[k], 5, 10.0)
            phi[k+1] = phi[k] + self.mu*dJ
        return phi, s


################################################################################
if __name__=="__main__":
    ordered_data = pickle.load(open("ordered_output.pkl", "rb"))
    # ordered_data = []
    # for k in range(44100):
    #     ordered_data.append(np.ones(2))
    #     ordered_data[k][0] = 525.0
    #     ordered_data[k][1] = 1060.0
    t = Template(510.0, 5)
    p, s = t.process_input(ordered_data)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax2 = ax1.twinx()
    ax1.plot(np.arange(len(p))/44100., p, color='k')
    ax2.plot(np.arange(len(p))/44100., s, color='r', linestyle='--')
    plt.show()
