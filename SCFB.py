import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
from FDL import FDL
from PLL import PLL

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
        pass

    def process_signal(self, in_sig):
        pass

    def plot_output(self, in_sig):
        pass

    def get_ordered_output(self, in_sig):
        pass


################################################################################
if __name__=="__main__":
    print("Still nothing!")
