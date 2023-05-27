from ops.operators import SingleOp
import jax.numpy as jnp

class Heaviside(SingleOp, op=lambda x: jnp.where(x>0, 1, 0)):
    pass