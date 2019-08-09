import numpy as np
from scipy.integrate import odeint

def dpendulum(state, t=0, pivot_x=0.0, pivot_y=0.0, is_acceleration=False, l=1.0, g=9.8, d=0.0, h=1e-4):
    """Returns the dynamical equation of a non inertial pendulum

    Parameters:
    state: the state (angle, angular speed)
    t: the time
    l: the pendulum's length
    g: the local acceleration of gravity
    d: the damping constant
    pivot_x: the horizontal position of the pivot
    pivot_y: the vertical position of the pivot
    is_acceleration: set to True to input pivot accelerations instead of positions
    h: numerical step for computing numerical derivatives

    Returns:
    dydt: the time derivative

    """

    ## Flexible input interpretation
    accel_x, accel_y = _format_accelerations(pivot_x, pivot_y, is_acceleration, h)

    ## Dynamical equation
    th, w = state
    dydt = [w,
            -g/l * np.sin(th)  - d * w - accel_x(t) * np.cos(th) / l  - accel_y(t) * np.sin(th) / l]

    return dydt

def pendulum(yinit, ts, pivot_x=0.0, pivot_y=0.0, is_acceleration=False, l=1.0, g=9.8, d=0.0, h=1e-4, **kwargs):
    """Returns the timeseries of a simulated non inertial pendulum

    Parameters:
    yinit: initial conditions (th, w)
    ts: integration times
    l: the pendulum's length
    g: the local acceleration of gravity
    d: the damping constant
    pivot_x: the horizontal position of the pivot
    pivot_y: the vertical position of the pivot
    is_acceleration: set to True to input pivot accelerations instead of positions
    h: numerical step for computing numerical derivatives
    **kwargs: odeint keyword arguments

    Returns:
    sol: the simulation's timeseries sol[:, 0] = ths, sol[:, 1] = ws

    """

    ## Set the problem
    f = lambda state, t : dpendulum(state, t, pivot_x, pivot_y, is_acceleration, l, g, d, h)

    ## Solve it
    sol = odeint(f, yinit, ts, **kwargs)

    return sol

def ddouble_pendulum(state, t=0, pivot_x=0.0, pivot_y=0.0, is_acceleration=False, m=(1, 1), l=(1,1), g=9.8, h=1e-4):
    """Returns the dynamical equation of a non-inertial double pendulum

    Parameters:
    state: the state (angle_1, angular speed_1, angle_2, angular_speed_2)
    t: the time
    m: the mass of each pendula
    l: the length of each pendula
    g: the local acceleration of gravity
    pivot_x: the horizontal position of the pivot
    pivot_y: the vertical position of the pivot
    is_acceleration: set to True to input pivot accelerations instead of positions
    h: numerical step for computing numerical derivatives

    Returns:
    dydt: the time derivative

    """

    ## Flexible input interpretation
    accel_x, accel_y = _format_accelerations(pivot_x, pivot_y, is_acceleration, h)

    ## Define some auxiliary variables
    M = np.sum(m)
    (l1, l2) = l
    (m1, m2) = m
    det = lambda th1, th2 : m2*(l1*l2)**2*(m1 + m2*np.sin(th1-th2)**2)
    a = lambda th1, th2 : m2*l2**2 / det(th1, th2)
    b = lambda th1, th2 : -m2*l1*l2*np.cos(th1 - th2) / det(th1, th2)
    d = lambda th1, th2 : M*l1**2 / det(th1, th2)

    mat = lambda th1, th2 : np.matrix(
                            [[1, 0,           0, 0],
                             [0, a(th1, th2), 0, b(th1, th2)],
                             [0, 0,           1, 0],
                             [0, b(th1, th2), 0, d(th1, th2)]]
                             )

    F = lambda th1, w1, th2, w2, t: np.matrix([[w1],
                                             [-m2*l1*l2*np.sin(th1-th2)*w2**2 - M*g*l1*np.sin(th1) - M*l1*(accel_x(t)*np.cos(th1) + accel_y(t)*np.sin(th1))],
                                             [w2],
                                             [m2*l1*l2*np.sin(th1-th2)*w1**2 - m2*g*l2*np.sin(th2) -m2*l2*(accel_x(t)*np.cos(th2) + accel_y(t)*np.sin(th2))]])

    ## Dynamical equations
    th1, w1, th2, w2 = state
    dydt = np.dot(mat(th1, th2), F(th1, w1, th2, w2, t))

    dydt = dydt.reshape(1,4).tolist()[0]
    return dydt

def double_pendulum(yinit, ts, pivot_x=0.0, pivot_y=0.0, is_acceleration=False, m=(1, 1), l=(1,1), g=9.8, h=1e-4, **kwargs):
    """Returns the timeseries of a simulated non-inertial double pendulum

    Parameters:
    yinit: initial conditions (th_1, w_1, th_2, w_2)
    ts: integration times
    m: the mass of each pendula
    l: the length of each pendula
    g: the local acceleration of gravity
    pivot_x: the horizontal position of the pivot
    pivot_y: the vertical position of the pivot
    is_acceleration: set to True to input pivot accelerations instead of positions
    h: numerical step for computing numerical derivatives

    Returns:
    sol: the simulation's timeseries
    sol[:, 0] = ths_1, sol[:, 1] = ws_1
    sol[:, 2] = ths_2, sol[:, 3] = ws_2
    """
    ## Set the problem
    f = lambda state, t : ddouble_pendulum(state, t, pivot_x, pivot_y, is_acceleration, m, l, g, h)

    ## Solve it
    sol = odeint(f, yinit, ts, **kwargs)

    return sol

def _format_accelerations(pivot_x, pivot_y, is_acceleration, h):
    """ Returns the pivot movement as acceleration

    The user is allowed to enter the pivot's movement as two functions of time.
    If is_acceleration is set to True, these functions are interpreted as
    pivot's acceleration. Otherwise, they are interpreted as pivot's movement.

    Parameters:
    pivot_x: the horizontal position of the pivot
    pivot_y: the vertical position of the pivot
    is_acceleration: set to True to input pivot accelerations instead of positions
    h: numerical step for computing numerical derivatives

    Returns:
    accel_x: the horizontal acceleration of the pivot, as a function of t
    accel_y: the vertical acceleration of the pivot, as a function of t
    """
    ## If the user introduces a constant, it should be interpreted as a function
    if not callable(pivot_x):
        value = pivot_x
        pivot_x = lambda t : value + 0.0*t

    if not callable(pivot_y):
        value = pivot_y
        pivot_y = lambda t : value + 0.0*t

    #
    if is_acceleration: # Just assign it
        accel_x = lambda t : pivot_x(t)
        accel_y = lambda t : pivot_y(t)
    else: # Compute the acceleration numerically
        speed_x = lambda t : (pivot_x(t + h) - pivot_x(t))/h
        speed_y = lambda t : (pivot_y(t + h) - pivot_y(t))/h
        accel_x = lambda t : (speed_x(t + h) - speed_x(t))/h
        accel_y = lambda t : (speed_y(t + h) - speed_y(t))/h
        #TODO: use a less artisanal method

    return accel_x, accel_y
