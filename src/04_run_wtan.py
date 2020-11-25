import numpy as np
import pickle
import matplotlib.pyplot as plt
import scfb
import scfbutils as su

templates = pickle.load(open('template_data.pkl', 'rb'))
t = np.arange(len(templates[0].strengths))*(1./44100)
strengths = [t.strengths for t in templates]
strengths = np.ascontiguousarray(np.flipud(np.stack(strengths, axis=0)))
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.imshow(strengths, aspect='auto')
print(strengths.shape)
k = np.ones((strengths.shape[0], strengths.shape[0]))*20.
# max_val = strengths.shape[0]*strengths.shape[1]
# for i in range(strengths.shape[0]):
#     for j in range(strengths.shape[0]):
#         k[i][j] = max_val/(max_val*(1 + abs(i-j))) + 2.
# k *= 5
tau = np.ones(strengths.shape[0])*0.001
wta_out = su.wta_net(strengths, k, strengths.shape[0], strengths.shape[1],
        1./44100, tau, 1., 2., 1.2)
print(wta_out.shape)
for k in range(wta_out.shape[0]):
    ax3.plot(t, wta_out[k,:])
ax2.imshow(wta_out, aspect='auto')
plt.show()
