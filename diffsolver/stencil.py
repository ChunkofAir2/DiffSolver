# future work on stencils
# hopefully be able to allow bcs to take in all kinds of stencils 
# and then we can rewrite the divergence and lapacian according to it
# that means 1D 5pt stencils, etc. :)
# from .ops.operators import Operator
import re
import jax.numpy as jnp

# example: (i-1) + (i+1) - 2*(i)
def pad(y, num=1):

    pass

def parse_stencil(mat, order=1):
    def stencil(t, y):
        grid = jnp.meshgrid(*(jnp.arange(i) for i in mat.shape))
        grid = [i.flatten() for i in grid]
        grid = jnp.stack(grid).transpose()
        centres = jnp.array([(i-1)//2 for i in mat.shape])
        for idx in grid:
            offset = idx - centres
            
            print(offset)
        pass
    return stencil

# class StencilBuilder:
#     pass

# class Stencil(Operator):
#     pass

mat = jnp.array([
    [[1, -2, 1]],
    [[1, -2, 1]],
    [[1, -2, 1]]
])
parse_stencil(mat)(0, jnp.array([1, 2, 3, 4]))
