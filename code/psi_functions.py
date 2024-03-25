import jax.numpy as jnp
from jax import jit

# psi(x) = (2*\pi*\cosh(0.5x)^{-1}
@jit
def psi_vgpmil(x):
    return (2.0 * jnp.pi * jnp.cosh(0.5 * x)) ** -1

@jit
def dif_psi_vgpmil(x):
    return -0.5 * jnp.tanh(0.5 * x) * psi_vgpmil(x)

# psi(x) \propto (beta + 0.5*x^2)^{-alpha}
# @partial(jit, static_argnums=(1,2,))
@jit
def psi_gvgpmil(x, alpha=1.0, beta=4.0):
    x_c = jnp.clip(x, -3.0, 3.0)
    return (beta + 0.5 * x_c ** 2) ** (-alpha)

@jit
def dif_psi_gvgpmil(x, alpha=1.0, beta=4.0):
    x_c = jnp.clip(x, -3.0, 3.0)
    return -alpha * (beta + 0.5 * x_c ** 2) ** (-alpha - 1) * x_c