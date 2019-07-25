## Import the required modules
from pendulum.models import *
import random
import matplotlib.pyplot as plt

## Set-up your problem
l = 1 # length
g = 9.8 # Gravity
d = 0 # Damping

ts = np.linspace(0, 10, 500) # Simulation time
yinit = (0, 1) # Initial condition (th_0, w_0)

ainit = 0 # Initial acceleration

dA = 0.01 # Delta of acceleration
accel_y = lambda t : 0.0*t

def pendulum_energy(y, l, g):
    """ Returns the energy of a simple (inertial) pendulum
    """

    (th, w) = y

    T = 0.5*(l*w)**2 # Kinetic
    V = g*l*(1 - np.cos(th)) # Potential

    return T + V

## Solve it
yprev = yinit
aprev = ainit
for i in range(0, len(ts)-2):
    accel_x_up = lambda t : aprev + dA + 0.0*t
    accel_x_eq = lambda t : aprev + 0.0*t
    accel_x_do = lambda t : aprev - dA + 0.0*t

    ## Update the states
    sol_up = ni_pendulum(yprev, ts[i:i+2], accel_x_up, accel_y, True, l=l, g=g, d=d)
    sol_eq = ni_pendulum(yprev, ts[i:i+2], accel_x_eq, accel_y, True, l=l, g=g, d=d)
    sol_do = ni_pendulum(yprev, ts[i:i+2], accel_x_do, accel_y, True, l=l, g=g, d=d)

    ynext_up = sol_up[:,-1]
    ynext_eq = sol_eq[:,-1]
    ynext_do = sol_do[:,-1]

    ## Calculate the energies
    energies = [pendulum_energy(ynext_up, l, g),
                pendulum_energy(ynext_eq, l, g),
                pendulum_energy(ynext_do, l, g)]

    ## Choose the optimal (and randomly in case of a draw)
    min_indices = np.where(energies == np.min(energies))[0]
    n_hits = len(min_indices)
    chosen_index = min_indices[random.randint(0, n_hits-1)]
    chosen_energy = energies[chosen_index]

    ## Choose next action
    if chosen_index==0:
        ynext = ynext_up
        anext = aprev + dA
    elif chosen_index==1:
        ynext = ynext_eq
        anext = aprev
    elif chosen_index==2:
        ynext = ynext_do
        anext = aprev - dA
    else:
        raise Exception('This state shpuld not be reachable')

    yprev = ynext
    aprev = anext

    print(f'Episode: {i}. Applied acceleration {anext:.2f}. Energy {chosen_energy:.2f}')
