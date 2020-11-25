import numpy as np
import scfbutils as su
import matplotlib.pyplot as plt

fs = 44100
dt = 1./fs
T = 1.
t = np.arange(0,T,dt)
e1 = np.ones_like(t)*70.0
e2 = np.ones_like(t)*72.8
e3 = np.ones_like(t)*79.8
e4 = np.ones_like(t)*85.0
e5 = np.ones_like(t)*95.8
E = np.stack([e1, e2, e3, e4, e5])
# E = np.flipud(E)
E = np.ascontiguousarray(E)
k = np.ones((5,5))*3.
tau = np.ones(5)*0.02
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

wta_out = su.wta_net(E, k, 5, len(t), dt, tau, 100., 3., 120.)
print(wta_out.shape)
print(E.shape)
for k in range(5):
    ax1.plot(t, E[k,:])
    ax2.plot(t, wta_out[k,:])
plt.show()
