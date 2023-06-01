# test the heat eqation

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from diffsolver.ops.lapacian import Laplacian
from diffsolver.tableau import ButcherTableau
from diffsolver.rk import ImplicitRungeKutta
from diffsolver.bc import NeumannBC
from diffsolver.ops.arithmatic import Neg

import jax.numpy as jnp
import jax

nbc = NeumannBC(Laplacian(1), [0, 0])
y1 = jnp.expand_dims(jnp.arange(0, 50, 10), 0).repeat(2, 0)
# y1 = jnp.expand_dims(jnp.expand_dims(jnp.arange(-10, 12, 2), 0).repeat(2, 0), 0).repeat(2, 0)
# tableau = ButcherTableau(
#     jnp.array([1/9, (16+jnp.sqrt(6))/36, (16-jnp.sqrt(6))/36]), 
#     jnp.array([0, (16-jnp.sqrt(6))/10, (16+jnp.sqrt(6))/10]), 
#     jnp.array(
#     [[1/9, (-1-jnp.sqrt(6))/18, (-1+jnp.sqrt(6))/18],
#     [1/9, (88+7*jnp.sqrt(6))/360, (88-43*jnp.sqrt(6))/360],
#     [1/9, (88-43*jnp.sqrt(6))/360, (88-7*jnp.sqrt(6))/360]])
# )
tableau = ButcherTableau(jnp.array([1/2, 1/2]), jnp.array([0, 1]), jnp.array([[0, 0],[1/2, 1/2]]))
rkno1 = ImplicitRungeKutta(y1, tableau, nbc, 0.05)

print(rkno1.y)
for i in range(int(600/0.05)):
# for i in range(15):
    rkno1.step()
print(rkno1.y)