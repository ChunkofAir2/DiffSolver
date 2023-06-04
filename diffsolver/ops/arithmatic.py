from .op_tys import CombineOp, SingleOp
import jax.numpy as jnp

# Very self explainatory

class Add(CombineOp, op=lambda x, y: jnp.add(x, y)):
    pass

class Multiply(CombineOp, op=lambda x, y: jnp.multiply(x, y)):
    pass

class Subtract(CombineOp, op=lambda x, y: jnp.subtract(x, y)):
    pass

class Divide(CombineOp, op=lambda x, y: jnp.divide(x, y)):
    pass

class Neg(SingleOp, op=lambda x: jnp.multiply(-1, x)):
    pass