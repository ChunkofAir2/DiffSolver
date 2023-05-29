# DiffSolver

DiffSolver is a small personal project to write a Method of Lines solver using JAX.
This project is in its infancy and it contains a lot of bugs and thing that need to
be ironed out. 

Example 1:
```python 
from diffsolver.ops.laplacian import Lapacian 
from diffsolver.rk import ImplicitRungeKutta
from diffsolver.tableau import ButcherTableau
from diffsolver.bc import NeumannBC

import matplotlib.pyplot as plt

import jax.numpy as jnp

fn = NeumannBC(Laplacian(1), [0, 0])
y1 = jnp.arange(15)
y1 = y1 = jnp.expand_dims(y1, 0).repeat(15, 0)

# Implicit Euler
tableau = ButcherTableau(jnp.array([1]), jnp.array([1]), jnp.array([1]))
rkno1 = ImplicitRungeKutta(y1, tableau, fn, 0.05)
plt.plot(rkno1.y)

for i in range(int(400/0.05)):
    rkno1.step()
    
plt.plot(rkno1.y)
```

