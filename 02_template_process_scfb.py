import numpy as np
import pickle
import matplotlib.pyplot as plt
import scfb 
import scfbutils

################################################################################
if __name__=="__main__":
    chunks, sig_len_n = pickle.load(open("chunks.pkl", "rb"))
    freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 100)
    # freqs = np.linspace(50.0, 4000.0, 100)
    num_h = 5
    sigma = 0.02 
    mu = 0.1
    temp_array = scfb.TemplateArray(chunks, sig_len_n, freqs, num_h, sigma, mu)
    temp_array.adapt()

    pickle.dump(temp_array.templates, open('template_data.pkl', 'wb'))
