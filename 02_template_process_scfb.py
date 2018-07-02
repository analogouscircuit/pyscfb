import numpy as np
import pickle
import matplotlib.pyplot as plt
import scfb 
import scfbutils

################################################################################
if __name__=="__main__":
    ordered_data = pickle.load(open("ordered_output.pkl", "rb"))
    # freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 100)
    freqs = np.linspace(50.0, 4000.0, 100)
    num_temps = len(freqs)
    templates = []
    num_harmonics = 5
    k = 0
    for f in freqs:
        print("Processing template {}/{}".format(k+1, num_temps))
        t = scfb.Template(f, num_harmonics, mu=0.5, sigma=20)
        t.process_input(ordered_data)
        templates.append(t)
        k += 1

    pickle.dump(templates, open('template_data.pkl', 'wb'))
