## Import the required modules
from pendulum.models import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.animation as animation

## Set-up your problem
g = 9.8 # Acceleration of gravity
l = 1 # Pendulum length
d = 1 # Damping

## Load data
data = pd.read_csv('./scripts/data.csv')

## Clean data
data = data[['t', 'x', 'y']] # Drop unneccessary info
data = data.sort_values(by=['t'])
data = data.dropna() # Remove artifacts

# Pivot's positions functions can be built interpolating the data
pos_x = interp1d(data.t, data.x, kind = 'cubic')
pos_y = interp1d(data.t, data.y, kind = 'cubic')

ts = np.linspace(-5, 10, 1000) # Simulation time
yinit = (0, 0) # Initial condition (th_0, w_0)

## Solve it
sol = pendulum(yinit, ts, pos_x, pos_y, g = g, l = l, d = d)

## Transform to cartesian coordinates
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
