import sys
import pickle
import scfb
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

# import wav file
print(str(sys.argv[1]))
f_name = str(sys.argv[1])
f_s, in_sig = scipy.io.wavfile.read(f_name)
in_sig = np.array(in_sig, dtype=np.float32)
in_sig = in_sig/(2**15)
in_sig = in_sig/np.max(in_sig)

# process through SCFB
peri = scfb.SCFB(100., 4000., 100, f_s)
peri.process_signal(in_sig, verbose=True)

# plot results
fig, ax = plt.subplots()
for fdl in peri.fdl:
    ax.plot(np.arange(0,len(in_sig))/f_s, fdl.f_record, linewidth=0.7,
            color='k', linestyle='--')
peri.plot_output(ax)
plt.show()

# write ordered data to file (for template processing)
pickle.dump(peri.get_ordered_output(), open("ordered_output.pkl", "wb"))

