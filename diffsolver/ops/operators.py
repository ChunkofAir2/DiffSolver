import jax.numpy as jnp
import jax
from copy import copy
from functools import partial
from typing import Dict
import inspect

class Operator:
    """
    General operator class

    Other operators extends this class. Some are obvious, for example 
    Addition, Subtraction, etc. Some are not so obvious, for example 
    the process of adding boundary conditions. The operators are then 
    taken by the Runge Kutta solver and used to solve the Diffeq. 

    Class Variables

    constant : `Object`
        the constant that is used by the class rather than an input
    """

    def __init__(self) -> None:
        self.constant = None

    @partial(jax.jit, static_argnums=(0, ))
    def __call__(self, t=None, y=None):
        if self.constant == None:
            return self.op(t, y)
        else :
            return self.op(t, self.constant)
    
    def set_y(self, y):
        self.constant = y

    def set_const(self, **kwargs):
        """Generally set the constant for both itself and its variables 
        
        This is a weird one because it sort of conflicts with the notion that 
        there's already a constant variable. However, there's an important distinction 
        between them. The "set_y" function sets the input to the `self.op` function, 
        while the constants are parameters like `dx` or `dt` that need to be modified 
        without being able to touch the original class (less you want to index into it)

        Parameters:
        
        **kwargs: in the form of `"param_name" = "new_value"` or
        `"param_name" = {"class_name": "new_value"}` 

        Notes:
        
        This isn't the best way to deal with constants, espcially because the names 
        could easily clash with one another. However, that is a problem for future me 
        to deal with. For now I'm happy with just coming up with different names instead.
        """
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