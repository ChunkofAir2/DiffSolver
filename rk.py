from functools import partial
from newton import newtons_method
from tableau import ButcherTableau

from ops.operators import Operator
from ops.lapacian import Laplacian

import jax.numpy as jnp
import jax.scipy.linalg as linalg
import jax 

from jax import config
config.update("jax_enable_x64", True)

class ImplicitRungeKutta():
    def __init__(self, y0, tableau, f: Operator, dt, solver=newtons_method):
        self.tableau = tableau
        self.f = jax.vmap(f, 0, 0)
        self.y = y0
        self.dt = dt
        self.is_inv = jnp.linalg.cond(self.tableau.A) < 1/jnp.finfo(jnp.float32).eps
        self.solver = solver
        
    @partial(jax.jit, static_argnums=(0,))
    def _get_next(self, y, z, dt):
        if self.tableau.A.ndim > 1:
            if self.is_inv:
                # if A is invertable, go the faster route
                A_inv = linalg.inv(self.tableau.A) 
                d = jnp.expand_dims(jnp.matmul(self.tableau.b, A_inv), 0).repeat(y.size, 0).reshape((-1, *z.shape[1:]))
                sums = jnp.sum(jax.vmap(jnp.multiply)(d, z), 0)
            else: 
                # if A is singular, have to evaluate f manually again !!!
                g = jnp.add(jnp.expand_dims(y, 0), z)
                feval = self.f(g)
                feval = jnp.moveaxis(feval, -1, 0)
                sums = jax.vmap(jnp.multiply, (0, 1), 0)(self.tableau.b, feval)
                sums = dt * sums.sum(0).T
        else:
            d = jnp.matmul(self.tableau.b, self.tableau.A).repeat(y.size).reshape((z.shape[0], -1))
            z = z.squeeze(-1)
            sums = jnp.multiply(d, z)
            sums = sums.reshape(y.shape)
        return y + sums
    
    def _step(self, y):
        z = self.solver(y, self.tableau, self.f, self.dt)
        y = self._get_next(y, z, self.dt)
        return y
    
    def step(self):
        self.y = self._step(self.y.copy())
        return self.y

def test():
    from bc import NeumannBC

    nbc = NeumannBC(Laplacian(1), [0, 0])
    y1 = jnp.expand_dims(jnp.arange(0, 5, dtype=jnp.float64), 0).repeat(2, 0)*3
    # tableau = ButcherTableau(
    #     jnp.array([1/9, (16+jnp.sqrt(6))/36, (16-jnp.sqrt(6))/36]), 
    #     jnp.array([0, (16-jnp.sqrt(6))/10, (16+jnp.sqrt(6))/10]), 
    #     jnp.array(
    #     [[1/9, (-1-jnp.sqrt(6))/18, (-1+jnp.sqrt(6))/18],
    #     [1/9, (88+7*jnp.sqrt(6))/360, (88-43*jnp.sqrt(6))/360],
    #     [1/9, (88-43*jnp.sqrt(6))/360, (88-7*jnp.sqrt(6))/360]])
    # )
    tableau = ButcherTableau(jnp.array([1/2, 1/2]), jnp.array([0, 1]), jnp.array([[0, 0],[1/2, 1/2]]))
    rkno1 = ImplicitRungeKutta(y1, tableau, nbc.op, 0.05)
    print(rkno1.y)
    for i in range(int(400/0.05)):
    # for i in range(100):
        rkno1.step()
    print(rkno1.y)

test()