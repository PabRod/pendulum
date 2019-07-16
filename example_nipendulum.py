## Import the required modules
from pendulum.models import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Set-up your problem
steps = 1000
ts = np.linspace(-5, 10, steps) # Simulation time
yinit = (0.1, 0) # Initial condition (th_0, w_0)
g = 9.8 # Acceleration of gravity
l = 1 # Pendulum length

pos_x = lambda t : np.arctan(5*t) # Pivot's position
pos_y = lambda t : 0*t

f = lambda state, t : ni_pendulum(state, t, pos_x, pos_y, is_acceleration = False, g = g, l = l, d = 1) # Dynamical equation as a function of (state, t)

## Solve it
sol = odeint(f, yinit, ts)

## Extract each coordinate
x_pivot = pos_x(ts) # Pivot's positions
y_pivot = pos_y(ts)
x = x_pivot + l*np.sin(sol[:, 0]) # Bob's positions
y = y_pivot - l*np.cos(sol[:, 0])

## Animate results
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', xlim=(-3, 3), ylim=(-3, 3))
ax.grid()

line, = ax.plot([], [], 'o-', lw=1)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [x[i], x_pivot[i]]
    thisy = [y[i], y_pivot[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (ts[i]))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(ts)),
                              interval=2, blit=True, init_func=init)


## Uncomment for saving
# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
#ani.save('im.mp4', writer = writer)

plt.show()
