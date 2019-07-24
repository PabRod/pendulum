import numpy as np
from scipy.integrate import odeint

def dpendulum(state, t=0, l=1, g=9.8, d=0):
    """Returns the dynamical equation of a simple pendulum

    Parameters:
    state: the state (angle, angular speed)
    t: the time
    l: the pendulum's length
    g: the local acceleration of gravity
    d: damping constant

    Returns:
    dydt: the time derivative

    """
    th, w = state
    dydt = [w,
            -g/l * np.sin(th) - d * w]

    return dydt

def pendulum(yinit, ts, l=1, g=9.8, d=0, **kwargs):
    """ Returns the timeseries of a simulated pendulum

    Parameters:
    yinit: initial conditions (th, w)
    ts: integration times
    l: the pendulum's length
    g: the local acceleration of gravity
    d: damping constant
    **kwargs: odeint keyword arguments

    Returns:
    sol: the simulation's timeseries sol[:, 0] = ths, sol[:, 1] = ws
    """

    ## Set the problem
    f = lambda state, t : dpendulum(state, t, l, g, d)

    ## Solve it
    sol = odeint(f, yinit, ts, **kwargs)

    return sol

def dni_pendulum(state, t, pivot_x, pivot_y, is_acceleration=False, l=1.0, g=9.8, d=0.0, h=1e-4):
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
    if is_acceleration:
        accel_x = lambda t : pivot_x(t)
        accel_y = lambda t : pivot_y(t)
    else: # Compute the acceleration numerically
        speed_x = lambda t : (pivot_x(t + h) - pivot_x(t))/h
        speed_y = lambda t : (pivot_y(t + h) - pivot_y(t))/h
        accel_x = lambda t : (speed_x(t + h) - speed_x(t))/h
        accel_y = lambda t : (speed_y(t + h) - speed_y(t))/h

    th, w = state
    dydt = [w,
            -g/l * np.sin(th)  - d * w - accel_x(t) * np.cos(th) / l  - accel_y(t) * np.sin(th) / l]

    return dydt

def ni_pendulum(yinit, ts, pivot_x, pivot_y, is_acceleration=False, l=1.0, g=9.8, d=0.0, h=1e-4, **kwargs):
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
    f = lambda state, t : dni_pendulum(state, t, pivot_x, pivot_y, is_acceleration, l, g, d, h)

    ## Solve it
    sol = odeint(f, yinit, ts, **kwargs)

    return sol

def ddouble_pendulum(state, t, m=(1, 1), l=(1,1), g=9.8):
    """Returns the dynamical equation of a double pendulum

    Parameters:
    state: the state (angle_1, angular speed_1, angle_2, angular_speed_2)
    t: the time
    m: the mass of each pendula
    l: the length of each pendula
    g: the local acceleration of gravity

    Returns:
    dydt: the time derivative

    """
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

    F = lambda th1, w1, th2, w2 : np.matrix([[w1],
                                             [-m2*l1*l2*np.sin(th1-th2)*w2**2 - M*g*l1*np.sin(th1)],
                                             [w2],
                                             [m2*l1*l2*np.sin(th1-th2)*w1**2 - m2*g*l2*np.sin(th2)]])

    th1, w1, th2, w2 = state
    dydt = np.dot(mat(th1, th2), F(th1, w1, th2, w2))

    dydt = dydt.reshape(1,4).tolist()[0]
    return dydt

def double_pendulum(yinit, ts, m=(1, 1), l=(1,1), g=9.8, **kwargs):
    """Returns the timeseries of a simulated double pendulum

    Parameters:
    yinit: initial conditions (th_1, w_1, th_2, w_2)
    ts: integration times
    m: the mass of each pendula
    l: the length of each pendula
    g: the local acceleration of gravity

    Returns:
    sol: the simulation's timeseries
    sol[:, 0] = ths_1, sol[:, 1] = ws_1
    sol[:, 2] = ths_2, sol[:, 3] = ws_2
    """
    ## Set the problem
    f = lambda state, t : ddouble_pendulum(state, t, m, l, g)

    ## Solve it
    sol = odeint(f, yinit, ts, **kwargs)

    return sol

def dni_double_pendulum(state, t, pivot_x, pivot_y, is_acceleration=False, m=(1, 1), l=(1,1), g=9.8, h=1e-4):
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
    if is_acceleration:
        accel_x = lambda t : pivot_x(t)
        accel_y = lambda t : pivot_y(t)
    else: # Compute the acceleration numerically
        speed_x = lambda t : (pivot_x(t + h) - pivot_x(t))/h
        speed_y = lambda t : (pivot_y(t + h) - pivot_y(t))/h
        accel_x = lambda t : (speed_x(t + h) - speed_x(t))/h
        accel_y = lambda t : (speed_y(t + h) - speed_y(t))/h

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

    th1, w1, th2, w2 = state
    dydt = np.dot(mat(th1, th2), F(th1, w1, th2, w2, t))

    dydt = dydt.reshape(1,4).tolist()[0]
    return dydt

def ni_double_pendulum(yinit, ts, pivot_x, pivot_y, is_acceleration=False, m=(1, 1), l=(1,1), g=9.8, h=1e-4, **kwargs):
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
    f = lambda state, t : dni_double_pendulum(state, t, pivot_x, pivot_y, is_acceleration, m, l, g, h)

    ## Solve it
    sol = odeint(f, yinit, ts, **kwargs)

    return sol
