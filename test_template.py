import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scfb 
import scfbutils



################################################################################
if __name__=="__main__":
    ordered_data = pickle.load(open("ordered_output.pkl", "rb"))
    # freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 100)
    freqs = np.linspace(50.0, 4000.0, 100)
    templates = []
    num_harmonics = 5
    for f in freqs:
        t = scfb.Template(f, num_harmonics, mu=0.5, sigma=20)
        t.process_input(ordered_data)
        templates.append(t)

    ## take a look at the templates and their spread on the frequency axis
    # fig, ax = plt.subplots()
    # f_axis = np.arange(20.0, 4000.0, 0.5)
    # for t in templates:
    #     ax.plot(f_axis, scfbutils.template_vals(f_axis, t.f0, t.sig, t.num_h),
    #             color='k', linewidth=0.5)

    
    # t = Template(510.0, 5)
    # p, s = t.process_input(ordered_data)

    ## 
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    lines = []
    for t in templates:
        ln, = plt.plot([t.f_vals[0], t.f_vals[0]], [0, t.strengths[0]])
        lines.append(ln)

    def init():
        ax.set_xlim(20.0, 4000.0)
        ax.set_ylim(0.0, 30.0)
        return lines,

    def update(k):
        for n in range(len(templates)):
            lines[n].set_data([templates[n].f_vals[k], templates[n].f_vals[k]],
                    [0, templates[n].strengths[k]])
        return lines,
        
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(ordered_data), 20, dtype=int),
            init_func=init, interval=.0001, repeat=False)

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
