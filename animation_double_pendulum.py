## Import the required modules
from pendulum.models import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Set-up your problem
m = (1, 1)
l = (1, 1)
ts = np.linspace(0, 100, 10000) # Simulation time
yinit = (np.pi/3, 0, np.pi/2, 0) # Initial condition (th_1, w_1, th_2, w_2)
f = lambda state, t : double_pendulum(state, t, m = m, l = l) # Dynamical equation as a function of (state, t)

# For using non-default parameters, use
#
# f = lambda state, t : pendulum(state, t, l = 2)

## Solve it
sol = odeint(f, yinit, ts)

## Extract each coordinate
x_1 = l[0]*np.sin(sol[:, 0]) # Bob's positions
y_1 = -l[0]*np.cos(sol[:, 0])
x_2 = x_1 + l[1]*np.sin(sol[:, 2])
y_2 = y_1 - l[1]*np.cos(sol[:, 2])

## Animate results
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', xlim=(-3, 3), ylim=(-3, 3))
ax.grid()

line_1, = ax.plot([], [], 'o-', lw=1)
line_2, = ax.plot([], [], 'o-', lw=1)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line_1.set_data([], [])
    line_2.set_data([], [])
    time_text.set_text('')
    return line_1, line_2, time_text


def animate(i):
    xs_1 = [x_1[i], 0.0]
    ys_1 = [y_1[i], 0.0]

    xs_2 = [x_2[i], x_1[i]]
    ys_2 = [y_2[i], y_1[i]]

    line_1.set_data(xs_1, ys_1)
    line_2.set_data(xs_2, ys_2)
    time_text.set_text(time_template % (ts[i]))
    return line_1, line_2, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(ts)),
                              interval=2, blit=True, init_func=init)


## Uncomment for saving
# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
#ani.save('im.mp4', writer = writer)

plt.show()
