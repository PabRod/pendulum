from pendulum.models import *
import pytest
import numpy as np

@pytest.mark.parametrize("input, exp_output", [
    ((0, 0), (0, 0)), # Stable equilibrium
    ((np.pi, 0), (0, 0))  # Unstable equilibrium
])
def test_pendulum(input, exp_output):

    tol = 1e-8

    df = pendulum(input)

    assert(df == pytest.approx(exp_output, tol)), \
        'pendulum is not behaving as expected'
