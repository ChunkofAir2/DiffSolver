from ops.operators import Operator
import jax.numpy as jnp
from copy import copy

# outer layer is for BCs 
def _1d_laplacian(y, dx, dim=0):
    r = 1 / (dx ** 2)
    side_shape = list(y.shape)
    side_shape[dim] = 1

    y = jnp.append(jnp.append(jnp.zeros(side_shape), y, axis=dim), jnp.zeros(side_shape), axis=dim)
    lefty = jnp.take(y, jnp.arange(0, y.shape[dim]-2), axis=dim)
    righty = jnp.take(y, jnp.arange(2, y.shape[dim]), axis=dim)
    centrey = jnp.take(y, jnp.arange(1, y.shape[dim]-1), axis=dim)

    return r * lefty + (-2 * r) * centrey + r * righty

class Laplacian(Operator):
    def __init__(self, dx):
        super().__init__()
        self.dx = dx

    def op(self, y):
        dx = copy(self.dx)

        if isinstance(dx, int) or isinstance(dx, float):
            dx = (dx, ) * y.ndim
        elif isinstance(dx, jnp.ndarray):
            assert dx.ndim == y.ndim

        dydt2 = jnp.zeros(y.shape)
        for dim, dxi in enumerate(dx):
            dydt2 += _1d_laplacian(y, dxi, dim)

        return dydt2
    
# y = jnp.expand_dims(jnp.array([0, 2, 4, 8, 9, 10, 11], dtype=jnp.float32), 0).repeat(4, 0)
# print(Laplacian(0.01).op(y))