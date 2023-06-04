from .operators import Operator
import jax.numpy as jnp
from copy import copy

def _1d_forward_diff(left, centre, right):
    return right - centre 

def _1d_backward_diff(left, centre, right):
    return centre - left 

def _1d_central_diff(left, centre, right):
    return (right - left)/2

def _1d_diff(y, dx, dim=0, type="central"):
    r = 1 / dx
    side_shape = list(y.shape)
    side_shape[dim] = 1

    y = jnp.append(jnp.append(jnp.zeros(side_shape), y, axis=dim), jnp.zeros(side_shape), axis=dim)
    left = jnp.take(y, jnp.arange(0, y.shape[dim]-2), axis=dim)
    right = jnp.take(y, jnp.arange(2, y.shape[dim]), axis=dim)
    centre = jnp.take(y, jnp.arange(1, y.shape[dim]-1), axis=dim)

    match type:
        case "central2":
            y = jnp.append(jnp.append(jnp.zeros(side_shape), y, axis=dim), jnp.zeros(side_shape), axis=dim)
            left2 = jnp.take(y, jnp.arange(0, y.shape[dim]-4), axis=dim)
            right2 = jnp.take(y, jnp.arange(4, y.shape[dim]), axis=dim)
            return (-right2 + 8*right - 8*left + left2)/12 * r
        case "central":
            return _1d_central_diff(left, centre, right) * r
        case "forward":
            return _1d_forward_diff(left, centre, right) * r
        case "backward":
            return _1d_backward_diff(left, centre, right) * r

class Divergence(Operator):
    def __init__(self, dx, mode="forward"):
        super().__init__()
        self.dx = dx
        self.mode=mode

    def op(self, t, y):
        dx = copy(self.dx)

        if isinstance(dx, int) or isinstance(dx, float):
            dx = (dx, ) * y.ndim
        elif isinstance(dx, jnp.ndarray):
            assert dx.ndim == y.ndim

        div = jnp.zeros(y.shape)
        for dim, dxi in enumerate(dx):
            div += _1d_diff(y, dxi, dim, type=self.mode)

        return div 

# y = jnp.expand_dims(jnp.array([0, 0, 4, 8, 9, 10, 10], dtype=jnp.float32), 0).repeat(4, 0)
# print(y)
# print(_1d_diff(y, 1, 1, "central2"))
# print(Divergence(1)(y=y))