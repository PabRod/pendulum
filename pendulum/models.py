import numpy as np
from scipy.integrate import odeint

def dpendulum(state, t=0, pivot_x=0.0, pivot_y=0.0, is_acceleration=False, l=1.0, g=9.8, d=0.0, h=1e-4):
    """Returns the dynamical equation of a non inertial pendulum

    :param state: the state (angle, angular speed)
    :param t: the time
    :param l: the pendulum's length
    :param g: the local acceleration of gravity
    :param d: the damping constant
    :param pivot_x: the horizontal position of the pivot
    :type pivot_x: function of time or constant
    :param pivot_y: the vertical position of the pivot
    :type pivot_y: function of time or constant
    :param is_acceleration: set to True to input pivot accelerations instead of positions
    :type is_acceleration: boolean
    :param h: numerical step for computing numerical derivatives
    :returns: the time derivative (dydt)

    """

    ## Flexible input interpretation
    accel_x, accel_y = _format_accelerations(pivot_x, pivot_y, is_acceleration, h)

    ## Dynamical equation (see drafts/Derivation ni_pendulum.pdf)
    th, w = state
    dydt = [w,
            -g/l * np.sin(th)  - d * w - accel_x(t) * np.cos(th) / l  - accel_y(t) * np.sin(th) / l]

    return dydt

def pendulum(yinit, ts, pivot_x=0.0, pivot_y=0.0, is_acceleration=False, l=1.0, g=9.8, d=0.0, h=1e-4, **kwargs):
    """Returns the timeseries of a simulated non inertial pendulum

    :param yinit: initial conditions (th, w)
    :param ts: integration times
    :param l: the pendulum's length
    :param g: the local acceleration of gravity
    :param d: the damping constant
    :param pivot_x: the horizontal position of the pivot
    :type pivot_x: function of time or constant
    :param pivot_y: the vertical position of the pivot
    :type pivot_y: function of time or constant
    :param is_acceleration: set to True to input pivot accelerations instead of positions
    :type is_acceleration: boolean
    :param h: numerical step for computing numerical derivatives
    :param ``**kwargs``: odeint keyword arguments
    :returns: the simulation's timeseries (sol[:, 0] = ths, sol[:, 1] = ws)

    """

    ## Avoid wrong inputs
    if (l <= 0.0): # Negative or zero lengths don't make sense
        raise ValueError('Wrong pendulum length (l). Expected positive float')

    if (d < 0.0): # A negative damping constant doesn't make sense
        raise ValueError('Wrong damping constant (d). Expected zero or positive float')

    if (len(yinit) != 2): # The initial conditions are (th_0, w_0). No more, and no less
        raise ValueError('Wrong initial condition (yinit). Expected 2-elements vector')

    if (h <= 0.0): # The numerical step for differentiation has to be positive
        raise ValueError('Wrong numerical step (h). Expected a positive float')

    ## Set the problem
    f = lambda state, t : dpendulum(state, t, pivot_x, pivot_y, is_acceleration, l, g, d, h)

    ## Solve it
    sol = odeint(f, yinit, ts, **kwargs)

    return sol

def ddouble_pendulum(state, t=0, pivot_x=0.0, pivot_y=0.0, is_acceleration=False, m=(1, 1), l=(1,1), g=9.8, h=1e-4):
    """Returns the dynamical equation of a non-inertial double pendulum

    :param state: the state (angle_1, angular speed_1, angle_2, angular_speed_2)
    :param t: the time
    :param m: the mass of each pendula
    :param l: the length of each pendula
    :param g: the local acceleration of gravity
    :param pivot_x: the horizontal position of the pivot
    :type pivot_x: function of time or constant
    :param pivot_y: the vertical position of the pivot
    :type pivot_y: function of time or constant
    :param is_acceleration: set to True to input pivot accelerations instead of positions
    :type is_acceleration: boolean
    :param h: numerical step for computing numerical derivatives
    :returns: the time derivative (dydt)

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

    mat = lambda th1, th2 : np.array(
                            [[1, 0,           0, 0],
                             [0, a(th1, th2), 0, b(th1, th2)],
                             [0, 0,           1, 0],
                             [0, b(th1, th2), 0, d(th1, th2)]]
                             )

    F = lambda th1, w1, th2, w2, t: np.array([[w1],
                                             [-m2*l1*l2*np.sin(th1-th2)*w2**2 - M*g*l1*np.sin(th1) - M*l1*(accel_x(t)*np.cos(th1) + accel_y(t)*np.sin(th1))],
                                             [w2],
                                             [m2*l1*l2*np.sin(th1-th2)*w1**2 - m2*g*l2*np.sin(th2) -m2*l2*(accel_x(t)*np.cos(th2) + accel_y(t)*np.sin(th2))]])

    ## Dynamical equations
    ## See (drafts/Derivation double_pendulum.pdf)
    th1, w1, th2, w2 = state
    dydt = np.dot(mat(th1, th2), F(th1, w1, th2, w2, t))

    dydt = dydt.reshape(1,4).tolist()[0]
    return dydt

def double_pendulum(yinit, ts, pivot_x=0.0, pivot_y=0.0, is_acceleration=False, m=(1, 1), l=(1,1), g=9.8, h=1e-4, **kwargs):
    """Returns the timeseries of a simulated non-inertial double pendulum

    :param yinit: initial conditions (th_1, w_1, th_2, w_2)
    :param ts: integration times
    :param m: the mass of each pendula
    :param l: the length of each pendula
    :param g: the local acceleration of gravity
    :param pivot_x: the horizontal position of the pivot
    :type pivot_x: function of time or constant
    :param pivot_y: the vertical position of the pivot
    :type pivot_y: function of time or constant
    :param is_acceleration: set to True to input pivot accelerations instead of positions
    :type is_acceleration: boolean
    :param h: numerical step for computing numerical derivatives
    :param ``**kwargs``: odeint keyword arguments
    :returns: sol: the simulation's timeseries (sol[:, 0] = ths_1, sol[:, 1] = ws_1, sol[:, 2] = ths_2, sol[:, 3] = ws_2)
    """

    ## Avoid wrong inputs
    if (np.min(m) <= 0.0) or (len(m) != 2): # Negative or zero masses don't make sense
        raise ValueError('Wrong pendulum masses (m). Expected 2 positive floats')

    if (np.min(l) <= 0.0) or (len(l) != 2): # Negative or zero lengths don't make sense
        raise ValueError('Wrong pendulum lengths (l). Expected 2 positive floats')

    if (len(yinit) != 4): # The initial conditions are (th_0, w_0, th_1, w_!). No more, and no less
        raise ValueError('Wrong initial condition (yinit). Expected 4-elements vector')

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

    :param pivot_x: the horizontal position of the pivot
    :type pivot_x: function of time or constant
    :param pivot_y: the vertical position of the pivot
    :type pivot_y: function of time or constant
    :param is_acceleration: set to True to input pivot accelerations instead of positions
    :type is_acceleration: boolean
    :param h: numerical step for computing numerical derivatives
    :returns: accel_x and accel_y, the horizontal and vertical accelerations of the pivot, as a function of t
    """

    ## Input interpretation
    if callable(pivot_x): # If the user inputs a function
        pass # Do nothing
    elif isinstance(pivot_x, float) or isinstance(pivot_x, int):
        # If the user introduces a constant, it should be interpreted as a function
        value_x = pivot_x
        pivot_x = lambda t : value_x + 0.0*t
    else:
        raise ValueError('Wrong horizontal pivot position. Use x = constant or x(t) = function of t')

    if callable(pivot_y): # If the user inputs a function
        pass # Do nothing
    elif isinstance(pivot_y, float) or isinstance(pivot_y, int):
        # If the user introduces a constant, it should be interpreted as a function
        value_y = pivot_y
        pivot_y = lambda t : value_y + 0.0*t
    else:
        raise ValueError('Wrong vertical pivot position. Use y = constant or y(t) = function of t')

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
