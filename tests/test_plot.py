import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from diffsolver import *

import jax.numpy as jnp
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# nbc = NeumannBC(Laplacian(1), [0, 0])
# y1 = jnp.expand_dims(jnp.expand_dims(jnp.arange(-10, 11, 2), 0).repeat(2, 0), 0).repeat(2, 0)


nbc = NDeriv(2, Laplacian(1))
# y1 = jnp.stack([jnp.arange(10), jnp.arange(10, 0, -1), jnp.arange(10, 0, -1), jnp.arange(10)])
y1 = jnp.expand_dims(jnp.append(jnp.zeros(60), jnp.arange(15)), 0).repeat(10, 0)
# y1 = jnp.zeros((11, 11)).at[5, 5].set(5)
y1 = nbc.get_nderiv_y(y1)
# tableau = ButcherTableau(jnp.array([1/2, 1/2]), jnp.array([0, 1]), jnp.array([[0, 0],[1/2, 1/2]]))
tableau = ButcherTableau(jnp.array([1]), jnp.array([1]), jnp.array([1]))
rkno1 = DirichletBC(ImplicitRungeKutta(y1, tableau, nbc, 0.05), [0, 0])

# rkno1.step()
# rkno1.step()

# print(rkno1.y)
# raise

gr = graph.graph.Grapher(rkno1, 20, 50, dt=0.05)
gr.run()
# print(gr.rk.y)
gr.plot_2d(dim=(1, ), cmap="hot")