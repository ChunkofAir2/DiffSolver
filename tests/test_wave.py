import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from diffsolver import *
from matplotlib import pyplot as plt

import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)

# op = NeumannBC(Laplacian(1), [0])
op = NDeriv(2, Laplacian(1))
tableau = ButcherTableau(
    jnp.array([1/2, 1/2]), 
    jnp.array([1/2 - 1/6*jnp.sqrt(3), 1/2 + 1/6*jnp.sqrt(3)]), 
    jnp.array([[1/4, 1/4 - (1/6)*jnp.sqrt(3)],
    [1/4 + (1/6)*jnp.sqrt(3), 1/4]])
)
# tableau = ButcherTableau(
#     jnp.array([1/2, 1/2]), 
#     jnp.array([1, 0]), 
#     jnp.array([[0, 0], [1/2, 1/2]])
# )
# tableau = ButcherTableau(jnp.array([1]), jnp.array([1/2]), jnp.array([1/2]))
# y1 = jnp.expand_dims(jnp.arange(0, 5, dtype=jnp.float64), 0).repeat(2, 0)
y1 = jnp.append(jnp.zeros(100), jnp.arange(33, dtype=jnp.float64)*0.2)
y1 = op.get_nderiv_y(y1)
# plt.plot(y1[-1])

rkno1 = DirichletBC(ImplicitRungeKutta(y1, tableau, op, 0.05), [0])
# rkno1.step()
# print(rkno1.y)
gr = graph.graph.Grapher(rkno1, rec_interval=30, end=200, dt=0.05)
gr.run()
gr.plot_1d((1, ))
# rkno1 = ImplicitRungeKutta(y1, tableau, op, 0.05)
# print(rkno1.y)
# rkno1.step()
# from diffsolver.newton import newtons_method
# x = newtons_method(y1, tableau, op, 0, 0.05)

# A_expand = jnp.expand_dims(tableau.A, -1).repeat(y1.size, -1).reshape((*tableau.A.shape, *y1.shape))
# f = 0.05 * jnp.multiply(A_expand, op(y=y1+x))
# if tableau.size > 1:
#     f = f.sum(1)
# print(f - x)

# for i in range(2400*3):
#     rkno1.step()
#     if i % (600) == 0 and i > 2400:
#         plt.plot(rkno1.y[-1])
# plt.show()
# print("op: ", op(y=rkno1.y))
# print("y: ", rkno1.y)


# plt.show()
# plt.plot(rkno1.y[-1])

# for i in range(int(400/0.05)):
# for i in range(3000):
#     rkno1.step()
#     if i * 0.01 % 3 == 0 and i*0.01 > 5:
#         plt.plot(rkno1.y[-1])
# plt.show()