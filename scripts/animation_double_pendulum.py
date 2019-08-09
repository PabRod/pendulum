## Import the required modules
from pendulum.models import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Set-up your problem
m = (2, 1) # Masses
l = (1, 1) # Lengths

## Pivot's position
pos_x = lambda t : -2 + np.arctan(3*t - 3*1) - np.arctan(3*t - 3*7)
pos_y = lambda t : 0.0*t

ts = np.linspace(0, 10, 1000) # Simulation time
yinit = (0, 0, 0, 0) # Initial condition (th_1, w_1, th_2, w_2)

## Solve it
sol = double_pendulum(yinit, ts, pos_x, pos_y, m=m, l=l)

## Extract each coordinate
x_0 = pos_x(ts) # Pivot's positions
y_0 = pos_y(ts)
x_1 = x_0 + l[0]*np.sin(sol[:, 0]) # Bob's positions
y_1 = y_0 - l[0]*np.cos(sol[:, 0])
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
    xs_1 = [x_1[i], x_0[i]]
    ys_1 = [y_1[i], y_0[i]]

    xs_2 = [x_2[i], x_1[i]]
    ys_2 = [y_2[i], y_1[i]]

    line_1.set_data(xs_1, ys_1)
    line_2.set_data(xs_2, ys_2)
    time_text.set_text(time_template % (ts[i]))
    return line_1, line_2, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(ts)),
                              interval=2, blit=True, init_func=init)


## Uncomment for saving
Writer = animation.writers['ffmpeg']
# writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('im.mp4', writer = writer)

plt.show()
