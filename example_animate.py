## Import the required modules
from pendulum.models import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Set-up your problem
steps = 1000
ts = np.linspace(0, 10, steps) # Simulation time
yinit = (0, 3) # Initial condition (th_0, w_0)
l = 1.5
f = lambda state, t : pendulum(state, t, l = l)

## Solve it
sol = odeint(f, yinit, ts)

## Extract each coordinate
x = l*np.sin(sol[:, 0])
y = -l*np.cos(sol[:, 0])
x_pivot = np.zeros(len(x))
y_pivot = np.zeros(len(y))

## Animate results
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', xlim=(-1.5*l, 1.5*l), ylim=(-1.5*l, 1.5*l))
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


plt.legend()
plt.show()
