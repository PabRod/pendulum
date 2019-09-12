## Import the required modules
from pendulum.models import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## Set-up your problem
g = 9.8 # Acceleration of gravity
l = 1 # Pendulum length
d = 1 # Damping

# Auxiliary function
# representing a continuous step
def cont_step(t, t_init, x_init, x_end, speed):
    aux = lambda t : np.pi/2 + np.arctan(speed*(t-t_init))/np.pi # Asymptotes: 0 and 1
    return x_init + (x_end - x_init) * aux(t)

# =============================================================================
# First simulation: one step to the right
tSteps = 1000
ts_1 = np.linspace(-1, 1, tSteps) # Simulation time
yinit_1 = (0, 0) # Initial condition (th_0, w_0)

pos_x = lambda t : cont_step(t, x_init = 0, x_end = 1, t_init = 0, speed = 5)
pos_y = lambda t : 0*t

# Solve
sol_1 = pendulum(yinit_1, ts_1, pos_x, pos_y, g = g, l = l, d = d)

# =============================================================================
# Second simulation: one step to the left
ts_2 = np.linspace(1, 3, tSteps) # Update the simulation time
yinit_2 = sol_1[-1,:] # Init with last state of previous simulation

# Rebuild step
# x_init is x_end of previous step
# x_end now represents a step leftwards
pos_x = lambda t : cont_step(t,  x_init = 1, x_end = 0, t_init = 2, speed = 5)
pos_y = lambda t : 0*t

# Solve
sol_2 = pendulum(yinit_2, ts_2, pos_x, pos_y, g = g, l = l, d = d)

# =============================================================================
# And so on...

## Plot results
fig, axs = plt.subplots(2, 1)
axs[0].plot(ts_1, sol_1[:,0], label = r'$\theta$')
axs[0].plot(ts_1, sol_1[:,1], label = r'$\omega$')

axs[0].set_title('First simulation')
axs[0].set_ylabel('states')
axs[0].set_xlim((-1, 3))
axs[0].set_ylim((-4, 4))

plt.legend()

axs[1].set_title('Second simulation')
axs[1].plot(ts_2, sol_2[:,0], label = r'$\theta$')
axs[1].plot(ts_2, sol_2[:,1], label = r'$\omega$')

axs[1].set_xlabel('time')
axs[1].set_ylabel('states')
axs[1].set_xlim((-1, 3))
axs[1].set_ylim((-4, 4))

plt.legend()

plt.show()

# TODO: write this in a loop
# TODO: paste the subsimulations into a single vector
# TODO: paste also the displacements of pivot_x
