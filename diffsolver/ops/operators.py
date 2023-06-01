import jax.numpy as jnp
import jax
from copy import copy
from functools import partial
from typing import Dict
import inspect

class Operator:
    def __init__(self) -> None:
        self.constant = None
        pass

    @partial(jax.jit, static_argnums=(0, ))
    def __call__(self, t=None, y=None):
        if self.constant == None:
            return self.op(t, y)
        else :
            return self.op(t, self.constant)
    
    def set_y(self, y):
        self.constant = y
    
    def set_const(self, **kwargs):
        for name, value in inspect.getmembers(self):
            if not name.startswith('_'):
                if issubclass(type(value), Operator):
                    value.set_const(**kwargs)
                elif not inspect.isfunction(value) and name in kwargs.keys():
                    if isinstance(kwargs[name], Dict):
                        if not type(self).__name__ in kwargs[name]:
                            raise ValueError(f"input {kwargs} does not contain {type(self).__name__}")
                        setattr(self, name, kwargs[name][type(self).__name__])
                    else :
                        setattr(self, name, kwargs[name])
    
        # raise NotImplementedError("need to implement \"set_const\" fn")

    def op(self, t, y):
        raise NotImplementedError("need to implement \"op\" fn")
    
class Variable(Operator):
    def op(self, t, y):
        return y

# class OperatorList:
#     def __init__(self, op_list):
#         self.op_list = op_list

#     @partial(jax.jit, static_argnums=(0, ))
#     def __call__(self, t=None, y=None):
#         rtn_list = []
#         if len(y) != len(self.op_list):
#             raise ValueError(f"len(y): {len(y)} does not equal len(op_list): {len(self.op_list)}")
        
#         for op, yi in zip(self.op_list, y):
#             rtn_list.append(op(t, yi))            
#         return rtn_list
    
#     def __getitem__(self, idx):
#         if isinstance(idx, int):
#             return self.op_list[idx]
    
# print(jax.jacfwd(Laplacian(0.01).op)(jnp.arange(0, 10, dtype=jnp.float32)))
# fn = Mult(Laplacian(0.01), Laplacian(0.01)).op

# fn = Square(Laplacian(0.01)).op
# print(fn(y))
# print(jax.jacfwd(Square(Laplacian(0.01)).op)())