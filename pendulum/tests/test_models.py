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
