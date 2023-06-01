from functools import partial
from .newton import newtons_method
from .tableau import ButcherTableau
from .ops.operators import Operator

import jax.numpy as jnp
import jax.scipy.linalg as linalg
import jax 

# from jax import config
# config.update("jax_enable_x64", True)
from typing import List, Tuple

class ImplicitRungeKutta():
    def __init__(self, y0, tableau: ButcherTableau, f: Operator, dt, solver=newtons_method):
        self.tableau = tableau
        self.f = jax.vmap(f, (0, 0), 0)
        self.y = y0
        self.dt = jnp.array(dt)
        if self.tableau.size > 1:
            self.is_inv = jnp.linalg.cond(self.tableau.A) < 1/jnp.finfo(jnp.float32).eps
        self.solver = solver
        self.t = jnp.array(0)
        
    @partial(jax.jit, static_argnums=(0, ))
    def _get_next(self, y, z, dt):
        if self.tableau.size > 1:
            if self.is_inv:
                # if A is invertable, go the faster route
                A_inv = linalg.inv(self.tableau.A) 
                d = jnp.expand_dims(jnp.matmul(self.tableau.b, A_inv), 0).repeat(y.size, 0).reshape((-1, *z.shape[1:]))
                sums = jnp.sum(jax.vmap(jnp.multiply)(d, z), 0)
            else: 
                # if A is singular, have to evaluate f manually again !!!
                g = jnp.add(jnp.expand_dims(y, 0), z)
                t = self.t + self.tableau.c * self.dt
                feval = self.f(t, g)
                # feval = jnp.moveaxis(feval, -1, 0)
                sums = jax.vmap(jnp.multiply, (0, 0), 0)(self.tableau.b, feval)
                sums = dt * sums.sum(0)
        else:
            d = jnp.matmul(self.tableau.b, self.tableau.A).repeat(y.size).reshape((-1, *z.shape[1:]))
            z = z.squeeze(0)
            sums = jnp.multiply(d, z)
            sums = sums.reshape(y.shape)
        return y + sums
    
    def _step(self, y):
        z = self.solver(y, self.tableau, self.f, self.t, self.dt)
        y = self._get_next(y, z, self.dt)
        return y
    
    def step(self):
        self.y = self._step(self.y)
        self.t += self.dt

class IRKList:
    def __init__(
        self, y0: Tuple[jnp.ndarray], 
        tableau: ButcherTableau, f: List[Operator], 
        dt: int, solver=newtons_method
    ):
        self.y0 = y0
        self.solvers = (ImplicitRungeKutta(y0i, tableau, fi, dt, solver) for i, y0i, fi in enumerate(zip(y0, f)))

    def step(self):
        for solver in self.solvers:
            solver.step()

    def __getitem__(self, i):
        return self.solvers[i]