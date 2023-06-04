# DiffSolver

DiffSolver is a small personal project to write a Method of Lines solver using JAX.
This project is in its infancy and it contains a lot of bugs and thing that need to
be ironed out. 

## Example 1:

```python 
import jax.numpy as jnp
from diffsolver import *

# Heat Eq (NeumannBC modifies the dy/dt)
fn = NeumannBC(Laplacian(1), [0, 0])
y1 = jnp.arange(15)
y1 = jnp.expand_dims(y1, 0).repeat(15, 0)

# Implicit Euler
tableau = ButcherTableau(jnp.array([1]), jnp.array([1]), jnp.array([1]))
rkno1 = ImplicitRungeKutta(y1, tableau, fn, 0.05)

# Grapher 
grp = grapher.grapher.Graph(rkno1, 50, 30, dt=0.05)
grp.run()
grp.plot_2d()
```

## Example 2 (Dirichlet Boundary Conditions)
```python 
import jax.numpy as jnp
from diffsolver import *

# Wave Eq
fn = NDeriv(2, Laplacian(1))
y1 = jnp.arange(15)

# Gauss-Legendre Order 4
tableau = ButcherTableau(
    jnp.array([1/2, 1/2]), 
    jnp.array([1/2 - 1/6*jnp.sqrt(3), 1/2 + 1/6*jnp.sqrt(3)]), 
    jnp.array([[1/4, 1/4 - (1/6)*jnp.sqrt(3)],
    [1/4 + (1/6)*jnp.sqrt(3), 1/4]])
)

# Dirichlet BC modifies the integrator
rkno1 = DirichletBC(ImplicitRungeKutta(y1, tableau, fn, 0.05), [0])

# Grapher 
grp = grapher.grapher.Graph(rkno1, 50, 30, dt=0.05)
grp.run()

# Choose the last index to plot
grp.plot_1d((-1, ))
```

## Known issues (to me):

1. The One-Way Wave Equation does not seem to work (very well); looking at if
it's a problem with the divergence implementation. (Beware)

2. Testing has not integrated PyTest yet. That is to be worked on. 

3. Lack of type annotations for functions and classes. (it's pretty low in my list of 
priorities though) Documentation is also lacking in all respects, but example 1 is basically 
everything there is to this library. 

4. Graphing does not support 3D heat maps. 

