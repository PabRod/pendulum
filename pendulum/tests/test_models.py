from pendulum.models import *
from pendulum.models import _format_accelerations
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

@pytest.mark.xfail(raises=ValueError)
def test_pendulum_wrong_length():
    ''' Test wrong input (length)
    '''
    ## Set the pendulum
    yinit = (0, 1)
    l = -1 # Wrong, negative length
    ts = np.linspace(0, 100, 100)

    ## This should raise an exception
    sol = pendulum(yinit, ts, l = l)

@pytest.mark.xfail(raises=ValueError)
def test_pendulum_wrong_damping():
    ''' Test wrong input (damping)
    '''
    ## Set the pendulum
    yinit = (0, 1)
    d = -1 # Wrong, negative damping
    ts = np.linspace(0, 100, 100)

    ## This should raise an exception
    sol = pendulum(yinit, ts, d = d)

@pytest.mark.xfail(raises=ValueError)
def test_pendulum_wrong_yinit():
    ''' Test wrong input (initial conditions)
    '''
    ## Set the pendulum
    yinit = (0, 1, 2) # Wrong, non 2D initial condition
    ts = np.linspace(0, 100, 100)

    ## This should raise an exception
    sol = pendulum(yinit, ts)

@pytest.mark.xfail(raises=ValueError)
def test_pendulum_wrong_h():
    ''' Test wrong input (step)
    '''
    ## Set the pendulum
    yinit = (0, 1)
    h = 0.0 # Wrong, non-positive step
    ts = np.linspace(0, 100, 100)

    ## This should raise an exception
    sol = pendulum(yinit, ts, h = h)

@pytest.mark.xfail(raises=ValueError)
def test_double_pendulum_wrong_l():
    ''' Test wrong input (lengths)
    '''
    ## Set-up your problem
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (np.pi/2, 0, np.pi/2, 0) # Initial condition

    l = (1, 2, 3)

    ## Solve it
    sol = double_pendulum(yinit, ts, l = l)

@pytest.mark.xfail(raises=ValueError)
def test_double_pendulum_wrong_m():
    ''' Test wrong input (masses)
    '''
    ## Set-up your problem
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (np.pi/2, 0, np.pi/2, 0) # Initial condition

    m = (-2, 2)

    ## Solve it
    sol = double_pendulum(yinit, ts, m = m)

@pytest.mark.xfail(raises=ValueError)
def test_double_pendulum_wrong_yinit():
    ''' Test wrong input (initial condition)
    '''
    ## Set-up your problem
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (np.pi/2, 0, np.pi/2) # Wrong, non-4D initial condition

    ## Solve it
    sol = double_pendulum(yinit, ts)

@pytest.mark.xfail(raises=ValueError)
def test_pendulum_wrong_xaccel():
    ''' Test wrong input (horizontal acceleration)
    '''
    ## Set-up your problem
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (np.pi/2, 0, np.pi/2) # Wrong, non-4D initial condition

    # Accelerations
    acc_x = (0, 0) # Wrong acceleration
    acc_y = lambda t: t

    sol = double_pendulum(yinit, ts, acc_x, acc_y)

@pytest.mark.xfail(raises=ValueError)
def test_pendulum_wrong_yaccel():
    ''' Test wrong input (vertical acceleration)
    '''
    ## Set-up your problem
    ts = np.linspace(0, 10, 1000) # Simulation time
    yinit = (np.pi/2, 0, np.pi/2) # Wrong, non-4D initial condition

    # Accelerations
    acc_x = lambda t: 0.1 # Wrong acceleration
    acc_y = lambda t: (t, t)

    sol = double_pendulum(yinit, ts, acc_x, acc_y)


def test_format_acceleration_differentiation():
    ''' Tests the pivot's movement and accelerations are clearly related
    '''
    tol = 1e-5
    h = 1e-4

    ## Input 1: constant accelerations
    pos_x = lambda t : -9.8/2*t**2
    pos_y = 0.0

    accel_x, accel_y = _format_accelerations(pos_x, pos_y, is_acceleration=False, h=h)

    expected_accel_x = -9.8
    assert(accel_x(0) == pytest.approx(expected_accel_x, tol)), \
        'The numerical differentiation is failing. Try decreasing h'

def test_format_acceleration_inputs():
    ''' Tests the pivot's movement and accelerations
    '''
    tol = 1e-5
    h = 1e-4

    ## Input 1: constant accelerations
    const_accel_x = 0.0
    const_accel_y = 1.0

    accel_x, accel_y = _format_accelerations(const_accel_x, const_accel_y, is_acceleration=True, h=h)
    assert(accel_x(0) == pytest.approx(const_accel_x, tol)), \
        'Constant accel_x is not being correctly interpreted'
    assert(accel_y(0) == pytest.approx(const_accel_y, tol)), \
        'Constant accel_y is not being correctly interpreted'

    ## Input 2: functional accelerations
    fun_accel_x = lambda t : 0.0
    fun_accel_y = lambda t : 1.0

    accel_x, accel_y = _format_accelerations(fun_accel_x, fun_accel_y, is_acceleration=True, h=h)
    assert(accel_x(0) == pytest.approx(const_accel_x, tol)), \
        'Functional accel_x is not being correctly interpreted'
    assert(accel_y(0) == pytest.approx(const_accel_y, tol)), \
        'Functional accel_y is not being correctly interpreted'

@pytest.mark.xfail(raises=ValueError)
def test_format_acceleration_errors():
    ''' Tests the exceptions for pivot's movement and accelerations
    '''
    tol = 1e-5
    h = 1e-4

    wrong_accel = (1, 2) # Wrong acceleration, too many dimensions

    accel_x, accel_y = _format_accelerations(wrong_accel, wrong_accel, is_acceleration=True, h=h)
