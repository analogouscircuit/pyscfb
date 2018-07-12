import numpy as np
import pickle
import matplotlib.pyplot as plt
import scfb
import scfbutils as su

templates = pickle.load(open('template_data.pkl', 'rb'))
strengths = [t.strengths for t in templates]
strengths = np.flipud(np.stack(strengths, axis=0))
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.imshow(strengths, aspect='auto')
print(strengths.shape)
k = np.ones((strengths.shape[0], strengths.shape[0]))*3.
tau = np.ones(strengths.shape[0])*0.02
wta_out = su.wta_net(strengths, k, strengths.shape[0], strengths.shape[1],
        1./44100, tau, 100., 2., 120.)
print(wta_out)
ax2.imshow(wta_out, aspect='auto')
plt.show()
