# boundary conditions... Yay!!! (actually it's painful at best but usually downright tear inducing)

import jax.numpy as jnp
from typing import List
from .ops.operators import Operator

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

def _apply_neumann_bound_1d(y, dydn=0, dim=0):
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
    dydns: List[jnp.ndarray]
    
    def __init__(self, one, dydns: List[jnp.ndarray]):
        super().__init__(one)
        self.dydns = [i if i != None else 0 for i in dydns]

    def apply_bound(self, y):
        # print("seen shape",  y.shape)
        assert y.ndim == len(self.dydns)
        return _apply_neumann_bound(y, self.dydns)
    
    def remove_bound(self, yb):
        return _remove_neumann_bound(yb)
    
# bc = NeumannBC(Laplacian(0.01), [0, 0])
# print(bc.op(jnp.expand_dims(jnp.arange(0, 5), 0).repeat(2, 0)))