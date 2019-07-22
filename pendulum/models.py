import numpy as np

def pendulum(state, t=0, l=1, g=9.8, d=0):
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

def ni_pendulum(state, t, pivot_x, pivot_y, is_acceleration=False, l=1.0, g=9.8, d=0.0, h=1e-4):
    """Returns the dynamical equation of a non inertial pendulum

    Parameters:
    state: the state (angle, angular speed)
    t: the time
    l: the pendulum's lenght
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

    th, w = state
    dydt = [w,
            -g/l * np.sin(th)  - d * w - accel_x(t) * np.cos(th) / l  - accel_y(t) * np.sin(th) / l]

    return dydt

def double_pendulum(state, t, m=(1, 1), l=(1,1), g=9.8):
    """Returns the dynamical equation of a double pendulum

    Parameters:
    state: the state (angle_1, angular speed_1, angle_2, angular_speed_2)
    t: the time
    m: the mass of each pendula
    l: the length of each pendula
    g: the local acceleration of gravity
    d: damping constant

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
                                             [-m2*l1*l2*np.sin(th1-th2)*w1**2 - m2*g*l2*np.sin(th2)]])

    th1, w1, th2, w2 = state
    dydt = np.dot(mat(th1, th2), F(th1, w1, th2, w2))

    dydt = dydt.reshape(1,4).tolist()[0]
    return dydt
