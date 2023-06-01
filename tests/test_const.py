# test navier stokes eqs 

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from diffsolver import *

import jax.numpy as jnp

nbc = NeumannBC(Laplacian(1), [0, 0])
y1 = jnp.expand_dims(jnp.arange(0, 5, dtype=jnp.float64), 0).repeat(2, 0)
# print(Add(nbc, Variable(), Variable())(y=y1))
# print(nbc(y=y1))
# li = OperatorList([NeumannBC(Laplacian(1), [0, 0]), Constant(init=jnp.array([10]))])
# tableau = ButcherTableau(jnp.array([1]), jnp.array([1]), jnp.array([1]))

# rkno1 = ImplicitRungeKutta([y1, 0], tableau, li, 0.01)
# print(rkno1.y)
# for i in range(100):
#     rkno1.step()
# print(rkno1.y)
# nbc.set_const(const_ldw=y1, dx=0.1)
# nbc.set_y(y1)

# print(nbc())