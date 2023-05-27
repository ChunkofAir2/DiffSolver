from jax import numpy as jnp 

class ButcherTableau:
    def __init__(self, b: jnp.ndarray, c: jnp.ndarray, A: jnp.ndarray):
        self.b = b
        self.c = c
        self.A = A
        
        if b.shape[0] != c.shape[0] or c.shape[0] != A.shape[0]:
            raise ValueError
            
        if len(A.shape) > 1 and A.shape[0] != A.shape[1]:
            raise ValueError
        
        self.size = A.shape[0]