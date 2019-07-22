## Import the required modules
from pendulum.models import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

## Set-up your problem
ts = np.linspace(0, 100, 10000) # Simulation time
yinit = (0, -1, 0, 1) # Initial condition (th_0, w_0)
f = double_pendulum # Dynamical equation as a function of (state, t)

# For using non-default parameters, use
#
# f = lambda state, t : pendulum(state, t, l = 2)

## Solve it
sol = odeint(f, yinit, ts)

## Plot results
fig, axs = plt.subplots(1, 1)
labels = [r'$\theta_1$', r'$\omega_1$', r'$\theta_2$', r'$\omega_2$']
show = [True, False, True, False]

for i in range(0, 4):
    if show[i]:
        plt.plot(ts, sol[:, i], label = labels[i])

plt.legend()
plt.show()
