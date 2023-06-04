import jax.numpy as jnp
from jax import flatten_util;
from jax import tree_map

# Doesn't really work in practice for some reason 
# so I'm just going to abandon it for now

# pls do not touch! 

def _rms_norm(x):
    x_sq = jnp.real(x * jnp.conj(x))
    return jnp.sqrt(jnp.mean(x_sq))

def rms_norm(x):
    x, _ = flatten_util.ravel_pytree(x)
    if x.size == 0:
        return 0
    return _rms_norm(x)

class PIDController:
    def __init__(self, p_beta=0, i_beta=1, d_beta=0, rtol=1e-3, atol=1e-6, safety=0.71):
        # copied from diffrax :)
        
        self.beta1 = (p_beta + i_beta + d_beta) / 3
        self.beta2 = -(p_beta + 2 * d_beta) / 3 
        self.beta3 = d_beta / 3
        
        self.atol, self.rtol = atol, rtol

        self.prev_ise = 0.01
        self.prev2_ise = 0.01
        self.prev_dt = 0.05
        
        self.safety = safety

    def scaled_error(self, y0, y1_candidate, ye):
        def _scale(_y0, _y1_candidate, _y_error):
            # In case the solver steps into a region for which the vector field isn't
            # defined.
            _nan = jnp.isnan(_y1_candidate).any()
            _y1_candidate = jnp.where(_nan, _y0, _y1_candidate)
            _y = jnp.maximum(jnp.abs(_y0), jnp.abs(_y1_candidate))
            return _y_error / (self.atol + _y * self.rtol)

        scaled_error = jnp.linalg.norm(tree_map(_scale, y0, y1_candidate, ye))

    def _ise(self, y, ye):
        scaled_e = jnp.linalg.norm(ye / (self.atol + y * self.rtol))
        inv_scaled_e = 1/scaled_e
        return inv_scaled_e

    def step(self, y, ye):
        ise = self._ise(y, ye)
        f1 = 1 if self.beta1 == 0 else (ise / 3)  ** self.beta1
        f2 = 1 if self.beta2 == 0 else (self.prev_ise / 3) ** self.beta2
        f3 = 1 if self.beta3 == 0 else (self.prev2_ise / 3) ** self.beta3
        
        (self.prev2_ise, self.prev_ise) = (self.prev_ise, ise)
        
        next_dt = jnp.clip(
            self.safety * f1 * f2 * f3,
            a_min=0.005,
            a_max=1.5,
        ) * self.prev_dt
        self.prev_dt = next_dt
        return next_dt