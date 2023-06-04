# boundary conditions... Yay!!! (actually it's painful at best but usually downright tear inducing)

import jax.numpy as jnp
from typing import List, Union

from .ops.operators import Operator
# from ops.lapacian import Laplacian

class BoundaryCond(Operator):
    def __init__(self, one: Operator):
        super().__init__()
        self.one = one

    def op(self, t, y):
        # print("op seen shape", y.shape)
        yb = self.apply_bound(y)
        yb = self.one(t, yb)
        y = self.remove_bound(yb)
        return y
    
    def apply_bound(self, y):
        raise NotImplementedError()
    
    def remove_bound(self, y):
        raise NotImplementedError()

def _apply_neumann_bound_1d(y, dydn: Union[int, float, List[jnp.ndarray]]=0, dim=0):
    new_shape = list(y.shape)
    new_shape[dim] = 1
    if isinstance(dydn, int) or isinstance(dydn, float) or dydn.size == 1:
        left = (jnp.take(y, 0, axis=dim) * (1 + dydn)).reshape(new_shape)
        right = (jnp.take(y, -1, axis=dim) * (1 + dydn)).reshape(new_shape)
        return jnp.append(jnp.append(left, y, axis=dim), right, axis=dim)
    
    # these don't really work so I won't bother/use them :)
    # elif dydn.size == jnp.product(jnp.array(new_shape)) * 2:
    #     left = (jnp.take(y, 0, axis=dim) * (1 + jnp.take(dydn, 0, axis=dim))).reshape(new_shape)
    #     right = (jnp.take(y, -1, axis=dim) * (1 + jnp.take(dydn, 1, axis=dim))).reshape(new_shape)
    #     return jnp.append(jnp.append(left, y, axis=dim), right, axis=dim)
    
def _apply_neumann_bound(y, dydns: List[jnp.ndarray]):
    for dim, dydn in enumerate(dydns):
        y = _apply_neumann_bound_1d(y, dydn, dim)
    return y

def _remove_neumann_bound(y):
    for dim in range(y.ndim):
        y = jnp.take(y, jnp.arange(1, y.shape[dim]-1), axis=dim)
    return y

class NeumannBC(BoundaryCond): 
    """Applies the Neumann Boundary Condition to another operator
    
    Usually used during the last step of building your differential operator.
    NB Cond are of the following: dy/dn = f(x) where x is on the boundary
    """

    dydns: List[jnp.ndarray]
    
    def __init__(self, one, dydns: List[jnp.ndarray]):
        super().__init__(one)
        self.dydns = [i if i != None else 0 for i in dydns]

    def apply_bound(self, y):
        # print("seen shape",  y.shape)
        if y.ndim != len(self.dydns):
            raise ValueError(f"the # of bc's: {self.dydns} does not match with the # of sides: {y.ndim} of the input")
        return _apply_neumann_bound(y, self.dydns)
    
    def remove_bound(self, yb):
        return _remove_neumann_bound(yb)

class DirichletBC:
    def __init__(self, rk, cond, dt=1, is_nderiv=True):
        self.rk = rk
        self.cond = cond
        self.dt = dt
        self.is_nderiv = is_nderiv

    def step(self):
        self.rk.step()

        if self.is_nderiv:
            self.rk.y = self.rk.y.at[-1].set(self.apply_bound(self.rk.y[-1]))
        else :
            self.rk.y = self.apply_bound(self.rk.y)
            
        self.y = self.rk.y
    
    def apply_bound(self, y):
        if y.ndim != len(self.cond):
            raise ValueError(f"the # of bc's: {self.cond} does not match with the # of sides: {y.ndim} of the input")

        for dim, c in enumerate(self.cond):
            if isinstance(c, tuple):
                dim = c[1]
                c = c[0]
            side_shape = list(y.shape)
            side_shape[dim] = 1
            new_y = jnp.take(y, jnp.arange(1, y.shape[dim]-1), dim)
            left = jnp.ones(side_shape) * c 
            right = jnp.ones(side_shape) * c 
            y = jnp.append(jnp.append(left, new_y, axis=dim), right, axis=dim)
        return y

    
# bc = DirichletBC(Laplacian(0.01), [(0, 1), (1, 0)])
# print(bc.op(None, jnp.expand_dims(jnp.arange(0, 5), 0).repeat(5, 0)))