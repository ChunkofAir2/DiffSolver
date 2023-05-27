import jax.numpy as jnp
import jax
from copy import copy

class Operator:
    def __init__(self) -> None:
        pass

    def __call__(self, t, y):
        return self.op(y)

    def op(self, y):
        raise NotImplementedError("need to implement \"op\" fn")

class CombineOp(Operator):
    one: Operator
    two: Operator

    def __init_subclass__(cls, op):
        super().__init__(cls)
        cls.double_op = lambda _, x, y: op(x, y)

    def __init__(self, one, two):
        self.one, self.two = one, two

    def op(self, y):
        one = self.one.op(y)
        two = self.two.op(y)
        return self.double_op(one, two)
    
class SingleOp(Operator):
    one: Operator

    def __init_subclass__(cls, op):
        super().__init__(cls)
        cls.single_op = lambda _, x: op(x)

    def __init__(self, one):
        self.one = one

    def op(self, y):
        one = self.one.op(y)
        return self.single_op(one)
    
class Square(SingleOp, op=lambda x: jnp.power(x, 2)):
    pass


    
# print(jax.jacfwd(Laplacian(0.01).op)(jnp.arange(0, 10, dtype=jnp.float32)))
# fn = Mult(Laplacian(0.01), Laplacian(0.01)).op

# fn = Square(Laplacian(0.01)).op
# print(fn(y))
# print(jax.jacfwd(Square(Laplacian(0.01)).op)())