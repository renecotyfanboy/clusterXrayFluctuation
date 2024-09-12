import jax
import numpy as np
import jax.numpy as jnp
from scipy.special import factorial
from functools import partial


def gen_Z_m(n):

    return np.arange(-n, n+1, 1)[np.arange(-n, n+1, 1)%2 == n%2]


def N(m, n):

    return jnp.sqrt(2*(n+1)+(1+m==0))

def R(rho, m, n):

    s = np.arange(0, (n-m)/2 + 1, 1)
    S = (-1)**s*factorial(n-s)/(factorial(s)*factorial((n+m)/2 -s)*factorial((n-m)/2 -s))

    return jnp.sum(S*(rho<1.)*rho**(n-2*s))


@partial(jnp.vectorize, excluded=(2, 3))
@partial(jax.jit,static_argnums=(2, 3))
def Z(rho, theta, m, n):

    assert abs(m) <= n
    assert (n-m)%2 == 0

    if m < 0:

        m = abs(m)
        return N(m, n)*R(rho, m, n)*jnp.sin(m*theta)

    else:

        return N(m, n)*R(rho, m, n)*jnp.cos(m*theta)

@jax.jit
def calc_Cz(sb, rad, ang):
    Cz = 0

    for n in range(9):

        for m in gen_Z_m(n):

            if m != 0:
                Cz += jnp.sqrt(
                    jnp.abs(jnp.nansum(sb * jnp.where(rad < 1, Z(rad, ang, m, n), jnp.nan) / jnp.pi, axis=(1, 2))))

    return Cz


def calc_G(sb, rad, ang):
    # A la Lovisari

    @jax.jit
    def g_func(x):
        mad = jnp.mean(jnp.abs(x[None, :] - x[:, None]))
        rmad = mad / jnp.mean(x)

        # Gini coefficient
        return 0.5 * rmad

    res = jnp.zeros((sb.shape[0],))

    for i in range(sb.shape[0]):
        ref_sb = jnp.where(rad[i] < 1., sb[i], jnp.nan)
        x = ref_sb[~jnp.isnan(ref_sb)]
        res = res.at[i].set(g_func(x))

    return res