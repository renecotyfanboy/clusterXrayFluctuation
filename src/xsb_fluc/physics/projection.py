import haiku as hk
import jax.numpy as jnp


def phi(t):
    return jnp.exp(jnp.pi / 2 * jnp.sinh(t))


def dphi(t):
    return jnp.pi / 2 * jnp.cosh(t) * jnp.exp(jnp.pi / 2 * jnp.sinh(t))


class AbelTransform(hk.Module):
    r"""
    Projection tool for 3D models. Relies on double exponential quadrature.

    $$\text{AbelTransform}\left\{ f\right\}(x, y) =  \int \text{d}z \, f(x, y, z)$$

    It uses double exponential quadrature to perform the integration.

    References :

    * [Takahasi and Mori (1974)](https://ems.press/journals/prims/articles/2686)
    * [Mori and Sugihara (2001)](https://doi.org/10.1016/S0377-0427(00)00501-X)
    * [Tanh-sinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature) from wikipedia

    """

    def __init__(self, model:"hk.Module", n_points=71):
        """
        Take a model and project it along the line of sight.

        Parameters:
            model (hk.Module): model to project
            n_points (int): number of points to use for the quadrature
        """
        super(AbelTransform, self).__init__()
        self.model = model
        self.n = n_points
        self.t = jnp.linspace(-3, 3, self.n)

    def __call__(self, r, *args):

        r = jnp.asarray(r)[..., None]
        t = self.t[None, :]
        x = phi(t)
        dx = dphi(t)

        return 2 * jnp.trapz(self.model(jnp.sqrt(r ** 2 + x ** 2), *args) * dx, x=t, axis=-1)
