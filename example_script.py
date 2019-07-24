## Import the required modules
from pendulum.models import *
import matplotlib.pyplot as plt

## Set-up your problem
l = 1 # length
g = 9.8 # Gravity
d = 1 # Damping

ts = np.linspace(0, 10, 1000) # Simulation time
yinit = (0, 1) # Initial condition (th_0, w_0)

## Solve it
sol = pendulum(yinit, ts, l, g, d)

## Plot results
fig, axs = plt.subplots(1, 1)
plt.plot(ts, sol[:,0], label = r'$\theta$')
plt.plot(ts, sol[:,1], label = r'$\omega$')

plt.legend()
plt.show()
