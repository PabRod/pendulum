from pendulum.models import *
from scipy.integrate import odeint
import numpy as np
import pytest

@pytest.mark.parametrize("input, exp_output", [
    ((0, 0), (0, 0)), # Stable equilibrium
    ((np.pi, 0), (0, 0))  # Unstable equilibrium
])
def test_pendulum(input, exp_output):
    ''' Test the equilibrium solutions
    '''
    tol = 1e-8

    df = pendulum(input)

    assert(df == pytest.approx(exp_output, tol)), \
        'pendulum is not behaving as expected'

def test_damped_pendulum():
    ''' Test the long-term solution of a damped pendulum
    '''
    tol = 1e-8

    ## Set the problem
    ts = np.linspace(0, 100, 100) # Simulation time
    yinit = (0, 1) # Initial condition (th_0, w_0)

    f = lambda state, t : pendulum(state, t, d = 2)

    ## Solve it
    sol = odeint(f, yinit, ts)

    last_theta = sol[-1, 0]
    last_w = sol[-1, 1]

    assert(last_theta == pytest.approx(0.0, tol))
    assert(last_w == pytest.approx(0.0, tol))

def test_undamped_pendulum():
    ''' Test the long-term solution of a damped pendulum
    '''
    tol = 1e-8

    ## Set the problem
    ts = np.linspace(0, 100, 100) # Simulation time
    yinit = (0, 1) # Initial condition (th_0, w_0)

    f = lambda state, t : pendulum(state, t, d = 0)

    ## Solve it
    sol = odeint(f, yinit, ts)

    last_theta = sol[-1, 0]
    last_w = sol[-1, 1]

    assert(last_theta != pytest.approx(0.0, tol))
    assert(last_w != pytest.approx(0.0, tol))

def test_freefall_pendulum():
    ''' Check the solution for a free-falling non-inertial pendulum
    '''
    tol = 1e-4

    ## Set-up your problem
    steps = 1000
    ts = np.linspace(0, 10, steps) # Simulation time
    yinit = (np.pi/2, 0) # Initial condition (th_0, w_0)
    g = 9.8 # Acceleration of gravity

    pos_x = lambda t : 0.0*t # Pivot's position
    pos_y = lambda t : -g/2*t**2 # Free falling

    f = lambda state, t : ni_pendulum(state, t, pos_x, pos_y, is_acceleration = False, g = g) # Dynamical equation as a function of (state, t)

    ## Solve it
    sol = odeint(f, yinit, ts)

    assert(sol[-1, 0] == pytest.approx(np.pi/2, tol))

def test_ni_pendulum_no_acceleration():
    ''' Tests the non inertial pendulum with no acceleration
    '''
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (0, 0) # Initial condition (th_0, w_0)
    forc_x = lambda t : 1.0*t # Uniform speed
    forc_y = lambda t : 2.0*t

    # The dynamics should be the same by virtue of Galileo's relativity
    f_inertial = lambda state, t : pendulum(state, t)
    f_non_intertial = lambda state, t : ni_pendulum(state, t, forc_x, forc_y)

    assert(f_inertial(yinit, 0.0) == f_non_intertial(yinit, 0.0))

def test_ni_pendulum():
    ''' Tests the non inertial pendulum with acceleration
    '''
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (0, 0) # Initial condition (th_0, w_0)
    forc_x = lambda t : 1.0*t**2 # Accelerated movement
    forc_y = lambda t : 2.0*t

    # The dynamics should be different
    f_inertial = lambda state, t : pendulum(state, t)
    f_non_intertial = lambda state, t : ni_pendulum(state, t, forc_x, forc_y)

    assert(f_inertial(yinit, 0.0) != f_non_intertial(yinit, 0.0))
