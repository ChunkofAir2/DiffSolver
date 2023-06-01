# an operator for converting nderivative diffeq into a series of diffeqs that are w.r.t. only dt

from .operators import Operator 
import jax.numpy as jnp

class NDeriv(Operator):
    def __init__(self, nderiv, inner):
        super().__init__()
        self.nderiv = nderiv
        self.inner = inner

    def get_nderiv_y(self, y):
        if isinstance(y, jnp.ndarray):
            new_y = [jnp.zeros(y.shape) for _ in range(self.nderiv-1)]
            new_y.append(y)

            return jnp.stack(new_y, axis=0)

    def op(self, t, y: jnp.ndarray):
        new_y = [self.inner(t, y[-1])]
        for i in range(0, y.shape[0]-1):
            new_y.append(y[i])

        return jnp.stack(new_y, axis=0)