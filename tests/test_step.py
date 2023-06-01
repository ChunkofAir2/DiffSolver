# test the heat eqation

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from diffsolver import *

import jax.numpy as jnp

nbc = NeumannBC(Laplacian(1), [0, 0])
y1 = jnp.expand_dims(jnp.arange(0, 5, dtype=jnp.float64), 0).repeat(2, 0)
tableau = ButcherTableau(jnp.array([1/2, 1/2]), jnp.array([0, 1]), jnp.array([[0, 0],[1/2, 1/2]]))
rkno1 = ImplicitRungeKutta(y1, tableau, nbc, 0.05)

print(rkno1.y)
# for i in range(int(400/0.05)):
for i in range(10):
    rkno1.step()
print(rkno1.y)