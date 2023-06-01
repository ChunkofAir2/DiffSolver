# newton's method solver for implicit runge kutta

from .tableau import ButcherTableau
from functools import partial

import jax 
from jax import flatten_util;
from jax import numpy as jnp

def jacdiag(f):
    def _jacdiag(x):
        def partial_grad_f_index(i):
            def partial_grad_f_x(xi):
                return f(x.at[i].set(xi))[i]
            return jax.grad(partial_grad_f_x)(x[i])
        return jax.vmap(partial_grad_f_index)(jax.numpy.arange(x.shape[0]))
    return _jacdiag

@partial(jax.jit, static_argnums=(0, ))
def jacnd(f, t, x):
    flat, unflatten = flatten_util.ravel_pytree(x)
    # print("unflattend_shape", unflatten(flat).shape)
    fp = lambda xi: flatten_util.ravel_pytree(f(t, unflatten(xi)))[0]
    return unflatten(jnp.diag(jax.jacfwd(fp)(flat)))

@partial(jax.jit, static_argnums=(1, 2))
def newtons_method(y: jnp.ndarray, tableau: ButcherTableau, F, t: jnp.ndarray, dt: jnp.ndarray):
    """calculate the newton's method """
    z = jnp.zeros((tableau.size, *y.shape))

    A_expand = jnp.expand_dims(tableau.A, 0).repeat(y.size, 0).reshape((*y.shape, *tableau.A.shape))
    t = t + tableau.c * dt

    # z_i' = 
    df = jacnd(F, t, jnp.add(jnp.expand_dims(y, 0), z))
    df = dt * jnp.multiply(A_expand, df.repeat(tableau.size).reshape(A_expand.shape))
    if tableau.size > 1:
        df = df.sum(-1)

    for i in range(3): 
        # f(z_j+y)
        temp_f = F(t, jnp.add(jnp.expand_dims(y, 0), z))

        # ref. z_i = h * sum_j(a_ij * f(y + z_j))
        f = dt * jnp.multiply(A_expand, temp_f.repeat(tableau.size).reshape(A_expand.shape))
        if tableau.size > 1:
            f = f.sum(-1)

        # solve for f(z) - z = 0
        # znext = z -(f-z)/(f-z)'
        f -= jnp.moveaxis(z, 0, -1)

        # because derivative of z w.r.t. z is one
        df -= 1

        diff = jnp.where(jnp.logical_and(f == 0, df == 0), 0, (f/df))
        z = z - jnp.moveaxis(diff, -1, 0)
    return z



def _test_():
    # tab = ButcherTableau(jnp.array([1]), jnp.array([1]), jnp.array([1]))
    tab = ButcherTableau(jnp.array([1/2, 1/2]), jnp.array([0, 1]), jnp.array([[0, 0],[1/2, 1/2]]))
    def lap(u) :
        uleft = jnp.append(jnp.ones((*u.shape[:-1], 1)) * jnp.expand_dims(u[..., 0], -1), u[..., :-1], -1)
        uright = jnp.append(u[..., 1:], jnp.ones((*u.shape[:-1], 1)) * jnp.expand_dims(u[..., -1], -1), -1)
        ucentre = u
        return uleft + uright - 2*ucentre
    test_arr = jnp.array(
        [[[0, 1, 2, 3], 
        [0, 1, 2, 3],
        [0, 1, 2, 3]], 
        [[0, 1, 2, 3], 
        [0, 1, 2, 3],
        [0, 1, 2, 3]]]
    )
    # print(newtons_method(tab, test_arr, lap, 0.01))
    print(newtons_method(tab, jnp.array([0, 1, 2, 3]), lap, 0.01))
    # print(jacdiag(f)(jnp.zeros([3])))

