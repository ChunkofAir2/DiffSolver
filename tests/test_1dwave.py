import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from diffsolver import *
import jax.numpy as jnp 

import matplotlib.pyplot as plt

from jax import config
config.update("jax_enable_x64", True)

op = NeumannBC(Neg(Divergence(1, mode="backward")), [0, -1])
def div(t=None, y=None):
    left = jnp.append(jnp.array([y[0]]), y[:-1])
    right = jnp.append(y[1:], jnp.array([y[-1]]))
    return right - y

y0 = jnp.expand_dims(jnp.append(jnp.arange(5, 0, -1), jnp.zeros(10)), 0).repeat(4, 0)
tableau = ButcherTableau(jnp.array([1/2, 1/2]), jnp.array([0, 1]), jnp.array([[0, 0],[1/2, 1/2]]))
# print(op(y=y0))
# print(div(y=y0))
# print(op(y=y0).shape)

rkno1 = ImplicitRungeKutta(y0, tableau, op, 0.05)
gr = graph.graph.Grapher(rkno1, 10, 20, dt=0.05)

# plt.plot(rkno1.y)
# for i in range(300):
#     rkno1.step()
#     if i % 60 == 0:
#         plt.plot(rkno1.y)
# plt.show()
gr.run()
gr.plot_2d(cmap="hot")