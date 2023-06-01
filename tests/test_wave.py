import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from diffsolver.ops.lapacian import Laplacian
from diffsolver.tableau import ButcherTableau
from diffsolver.rk import ImplicitRungeKutta
from diffsolver.bc import NeumannBC
from diffsolver.ops.nderiv import NDeriv

import jax.numpy as jnp

op = NDeriv(2, NeumannBC(Laplacian(1), [0, 0]))
tableau = ButcherTableau(jnp.array([1/2, 1/2]), jnp.array([0, 1]), jnp.array([[0, 0],[1/2, 1/2]]))
# y1 = jnp.expand_dims(jnp.arange(0, 5, dtype=jnp.float64), 0).repeat(2, 0)
y1 = jnp.expand_dims(jnp.arange(10), 0).repeat(2, 0)
y1 = op.get_nderiv_y(y1)
print(y1.shape)
rkno1 = ImplicitRungeKutta(y1, tableau, op, 0.01)
print(rkno1.y)

# for i in range(int(400/0.05)):
for i in range(200):
    rkno1.step()
print(rkno1.y)