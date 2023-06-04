# an operator for converting nderivative diffeq into a series of diffeqs that are w.r.t. only dt

from .operators import Operator 
import jax.numpy as jnp

class NDeriv(Operator):
    """ 
    Take an Operator that would be interpreted as a 
    1st order diffeq into a nth order diffeq

    This is usually to encapsulate an operator otherwise meant for a 
    diffeq of the form: `dy/dt = f(t, x, y)` into the form: `d^(n)y/dt^(n) = f(t, x, y)`. 
    It does this by adding more parameters to store the values of `d^(n-1)y/dt^(n-1)`
    until `n-1 = 1`. 
    """

    def __init__(self, nderiv, inner):
        """ Initialise NDeriv, see description of the class 

        Parameters:
            nderiv: the order of the diffeq
            nner: the operator or f(t, x, y)
        """
        super().__init__()
        self.nderiv = nderiv
        self.inner = inner

    def get_nderiv_y(self, y):
        """ Turn y0 into the form that this operator takes in 

        Parameters:
            y: y0
        """
        if isinstance(y, jnp.ndarray):
            new_y = [jnp.zeros(y.shape) for _ in range(self.nderiv-1)]
            new_y.append(y)

            return jnp.stack(new_y, axis=0)

    def op(self, t, y: jnp.ndarray):
        new_y = [self.inner(t, y[-1])]
        for i in range(0, self.nderiv-1):
            new_y.append(y[i])

        return jnp.stack(new_y, axis=0)