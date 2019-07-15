import numpy as np

def pendulum(state, t=0, l=1, g=9.8):
    """Returns the dynamical equation of a simple pendulum

    Parameters:
    state: the state (angle, angular speed)
    t: the time
    l: the pendulum's lenght
    g: the local acceleration of gravity

    Returns:
    dydt: the time derivative

   """
    th, w = state
    dydt = [w,
            -g/l * np.sin(th)]

    return dydt
