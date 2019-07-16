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
            -g/l * np.sin(th)  - d * w - accel_x(t) * np.cos(th) - accel_y(t) * np.sin(th)]

    return dydt
