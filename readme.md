[![Build Status](https://travis-ci.com/PabRod/pendulum.svg?branch=master)](https://travis-ci.com/PabRod/pendulum)
[![Code coverage](https://codecov.io/gh/PabRod/pendulum/graph/badge.svg)](https://codecov.io/gh/PabRod/pendulum)
[![codecov](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Pendulum simulator
Mechanical simulation of non-inertial pendula.

By [Pablo Rodríguez-Sánchez](https://pabrod.github.io) [![](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-2855-940X)

## Non-inertial pendula

|                            |                         |
|:--------------------------:|:-----------------------:|
| ![](figs/displacement.gif) | ![](figs/slingshot.gif) |

## Double pendulum (inertial and non-inertial)

|                      |                        |
|:--------------------:|:----------------------:|
| ![](figs/double.gif) | ![](figs/nidouble.gif) |

## Damped pendulum
![](figs/damped.gif)

# Installation
```
git clone https://github.com/PabRod/pendulum
cd pendulum
python setup.py install --user
```

# Tests
We are using `pytest` for unit testing. Run it via:

```
pytest
```

# Getting started

## Tutorial
Take a look at [our tutorial]('vignettes/tutorial.html').

Printer friendly version available [here]('vignettes/tutorial.pdf')

## Minimal example

This is a minimal example of the usage of this package:

```python
## Import the required modules
from pendulum.models import *
import matplotlib.pyplot as plt

## Set-up your problem
l = 1.5 # Length
g = 9.8 # Gravity
d = 0.5 # Damping

ts = np.linspace(0, 10, 1000) # Simulation time
yinit = (0, 1) # Initial condition (th_0, w_0)

## Solve it
sol = pendulum(yinit, ts, l = l, g = g, d = d)

## Plot results
fig, axs = plt.subplots(1, 1)
plt.plot(ts, sol[:,0], label = r'$\theta$')
plt.plot(ts, sol[:,1], label = r'$\omega$')

plt.xlabel('time')
plt.ylabel('states')

plt.legend()
plt.show()
```

## More examples
For more advanced examples, see

- [Simple pendulum](scripts/example_script.py)
- [Double pendulum](scripts/example_double_pendulum.py)
- [Non-inertial simple pendulum (animated)](scripts/animation_nipendulum.py)
- [Non-inertial double pendulum (animated)](scripts/animation_double_pendulum.py)
- [Reading pivot's position from data](scripts/animation_nipendulum_interp.py)
