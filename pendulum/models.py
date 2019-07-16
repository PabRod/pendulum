import numpy as np

def pendulum(state, t=0, l=1, g=9.8, d=0):
    """Returns the dynamical equation of a simple pendulum

    Parameters:
    state: the state (angle, angular speed)
    t: the time
    l: the pendulum's lenght
    g: the local acceleration of gravity
    d: damping constant

    Returns:
    dydt: the time derivative

   """
    th, w = state
    dydt = [w,
            -g/l * np.sin(th) - d * w]

    return dydt
