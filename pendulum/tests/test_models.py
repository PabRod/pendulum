from pendulum.models import *
import numpy as np
import pytest

@pytest.mark.parametrize("input, exp_output", [
    ((0, 0), (0, 0)), # Stable equilibrium
    ((np.pi, 0), (0, 0))  # Unstable equilibrium
])
def test_dpendulum(input, exp_output):
    ''' Test the equilibrium solutions
    '''
    tol = 1e-8

    df = dpendulum(input)

    assert(df == pytest.approx(exp_output, tol)), \
        'pendulum is not behaving as expected'

def test_damped_pendulum():
    ''' Test the long-term solution of a damped pendulum
    '''
    tol = 1e-8

    ## Set the pendulum
    yinit = (0, 1)
    d = 2 # Damping
    ts = np.linspace(0, 100, 100)
    sol = pendulum(yinit, ts, d = d)

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

    ## Solve it
    sol = pendulum(yinit, ts)

    last_theta = sol[-1, 0]
    last_w = sol[-1, 1]

    assert(last_theta != pytest.approx(0.0, tol))
    assert(last_w != pytest.approx(0.0, tol))

def test_freefall_pendulum():
    ''' Check the solution for a free-falling non-inertial pendulum
    '''
    tol = 1e-4

    ## Set-up your problem
    g = 9.8 # Acceleration of gravity
    pos_x = lambda t : 0.0*t # Pivot's position
    pos_y = lambda t : -g/2*t**2 # Free falling

    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (np.pi/2, 0) # Initial condition (th_0, w_0)

    ## Solve it
    sol = pendulum(yinit, ts, pos_x, pos_y, g = g)

    ## No relative movement is expected
    assert(sol[-1, 0] == pytest.approx(yinit[0], tol))

    # Repeat test in acceleration mode
    acc_x = lambda t: 0.0*t # Pivot's acceleration
    acc_y = lambda t: 0.0*t - g

    sol_2 = pendulum(yinit, ts, acc_x, acc_y, is_acceleration = True, g = g)

    ## No relative movement is expected
    assert(sol_2[-1, 0] == pytest.approx(yinit[0], tol))

def test_noninertial_pendulum_no_acceleration():
    ''' Tests the non inertial pendulum with no acceleration
    '''
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (0, 0) # Initial condition (th_0, w_0)
    forc_x = lambda t : 1.0*t # Uniform speed
    forc_y = lambda t : 2.0*t

    # The dynamics should be the same by virtue of Galileo's relativity
    f_inertial = lambda state, t : dpendulum(state, t)
    f_non_intertial = lambda state, t : dpendulum(state, t, forc_x, forc_y)

    assert(f_inertial(yinit, 0.0) == f_non_intertial(yinit, 0.0))

def test_noninertial_pendulum():
    ''' Tests the non inertial pendulum with acceleration
    '''
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (0, 0) # Initial condition (th_0, w_0)
    forc_x = lambda t : 1.0*t**2 # Accelerated movement
    forc_y = lambda t : 2.0*t

    # The dynamics should be different
    f_inertial = lambda state, t : dpendulum(state, t)
    f_non_intertial = lambda state, t : dpendulum(state, t, forc_x, forc_y)

    assert(f_inertial(yinit, 0.0) != f_non_intertial(yinit, 0.0))

@pytest.mark.parametrize("input, exp_output", [
    ((0, 0, 0, 0), (0, 0, 0, 0)), # Stable equilibrium
    ((np.pi, 0, 0, 0), (0, 0, 0, 0)),  # Unstable equilibria
    ((0, 0, np.pi, 0), (0, 0, 0, 0)),
    ((np.pi, 0, np.pi, 0), (0, 0, 0, 0))
])
def test_ddouble_pendulum(input, exp_output):
    ''' Test the equilibrium solutions
    '''
    tol = 1e-8

    df = ddouble_pendulum(input, 0)

    assert(df == pytest.approx(exp_output, tol)), \
        'pendulum is not behaving as expected'

def test_ni_double_pendulum_no_acceleration():
    '''Tests the non-inertial double pendulum with no acceleration
    '''
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (0, 0, 0, 0) # Initial condition (th_0, w_0, th_1, w_1)
    forc_x = lambda t : 1.0*t # Uniform speed
    forc_y = lambda t : 2.0*t

    # The dynamics should be the same by virtue of Galileo's relativity principle
    f_inertial = lambda state, t : ddouble_pendulum(state, t)
    f_non_intertial = lambda state, t : ddouble_pendulum(state, t, forc_x, forc_y)

    assert(f_inertial(yinit, 0.0) == f_non_intertial(yinit, 0.0))

def test_freefall_double_pendulum():
    ''' Check the solution for a free-falling non-inertial pendulum
    '''
    tol = 1e-4

    ## Set-up your problem
    g = 9.8 # Acceleration of gravity
    pos_x = lambda t : 0.0*t # Pivot's position
    pos_y = lambda t : -g/2*t**2 # Free falling

    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (np.pi/2, 0, np.pi/2, 0) # Initial condition (th_0, w_0)

    ## Solve it
    sol = double_pendulum(yinit, ts, pos_x, pos_y, g = g)

    ## No relative movement is expected
    assert(sol[-1, 0] == pytest.approx(yinit[0], tol))

    # Repeat test in acceleration mode
    acc_x = lambda t: 0.0*t # Pivot's acceleration
    acc_y = lambda t: 0.0*t - g

    sol_2 = double_pendulum(yinit, ts, acc_x, acc_y, is_acceleration = True, g = g)

    ## No relative movement is expected
    assert(sol_2[-1, 0] == pytest.approx(yinit[0], tol))
