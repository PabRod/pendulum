## Import the required modules
from pendulum.models import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

## Set-up your problem
ts = np.linspace(0, 10, 1000) # Simulation time
yinit = (0, 1) # Initial condition (th_0, w_0)
f = pendulum # Dynamical equation as a function of (state, t)

# For using non-default parameters, use
#
# f = lambda state, t : pendulum(state, t, l = 2)

## Solve it
sol = odeint(f, yinit, ts)

## Plot results
fig, axs = plt.subplots(1, 1)
plt.plot(ts, sol[:,0], label = r'$\theta$')
plt.plot(ts, sol[:,1], label = r'$\omega$')

plt.legend()
plt.show()
