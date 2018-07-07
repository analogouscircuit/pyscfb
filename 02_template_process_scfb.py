import numpy as np
import pickle
import matplotlib.pyplot as plt
import scfb 
import scfbutils

################################################################################
if __name__=="__main__":
    ordered_data = pickle.load(open("ordered_output.pkl", "rb"))
    chunks, sig_len_n = pickle.load(open("chunks.pkl", "rb"))
    td = scfbutils.TemplateData(chunks, sig_len_n)
    # freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 100)
    freqs = np.linspace(50.0, 4000.0, 100)
    # num_temps = len(freqs)
    # templates = []
    # num_harmonics = 5
    # k = 0
    # for f in freqs:
    #     print("Processing template {}/{}".format(k+1, num_temps))
    #     t = scfb.Template(f, num_harmonics, mu=0.5, sigma=20)
    #     # t.process_input(ordered_data)
    #     t.process_chunks(chunks, sig_len_n)
    #     templates.append(t)
    #     k += 1
    num_h = 5
    sigma = 10.0
    mu = 0.1
    temp_array = scfb.TemplateArray(chunks, sig_len_n, freqs, num_h, sigma, mu)
    temp_array.adapt()

    # pickle.dump(templates, open('template_data.pkl', 'wb'))
    pickle.dump(temp_array.templates, open('template_data.pkl', 'wb'))
