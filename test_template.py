import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scfb import Template



################################################################################
if __name__=="__main__":
    ordered_data = pickle.load(open("ordered_output.pkl", "rb"))
    t = Template(510.0, 5)
    p, s = t.process_input(ordered_data)

    ## Simple stem plto of a time step
    fig, ax = plt.subplots()
    # ln, = plt.plot([p[0],p[0]],[0, s[0]],animated=True)
    ln, = plt.plot([p[0],p[0]],[0, s[0]])

    def init():
        ax.set_xlim(20.0, 4000.0)
        ax.set_ylim(0.0, 5.0)
        return ln,

    def update(k):
        ln.set_data([p[k], p[k]], [0, s[k]])
        return ln,
        
    ani = FuncAnimation(fig, update, frames=np.arange(len(p),dtype=int),
            init_func=init, interval=.0005, repeat=False)

    ## Animation
    # fig, ax = plt.subplots()
    # def init():

    ## Regular plot (time on x-axis)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,1,1)
    # ax2 = ax1.twinx()
    # ax1.plot(np.arange(len(p))/44100., p, color='k')
    # ax2.plot(np.arange(len(p))/44100., s, color='r', linestyle='--')

    plt.show()
