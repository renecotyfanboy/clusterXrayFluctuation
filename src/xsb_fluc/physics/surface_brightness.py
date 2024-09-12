import haiku as hk
import astropy.units as u
import jax.numpy as jnp
from jax.scipy.special import gammaln
from haiku.initializers import Constant
from .projection import AbelTransform
from .emissivity import XrayEmissivity
from ..data.cluster import Cluster


class XraySurfaceBrightness(hk.Module):
    """
    Xray surface brightness model from 3D emissivity
    """

    def __init__(self, redshift, r_500):
        super(XraySurfaceBrightness, self).__init__()
        self.emissivity = XrayEmissivity(redshift)
        self.surface_brightness = AbelTransform(self.emissivity)
        self.r_500 = r_500.to(u.kpc).value

    @classmethod
    def from_data(cls, data: "Cluster"):
        """
        Create a surface brightness model from a `Cluster` object
        """

        return cls(data.z, data.r_500)

    def __call__(self, r, nh):
        log_bkg = hk.get_parameter("log_bkg", [], init=Constant(-5.))

        # As we integrate toward the l.o.s in r_500 units,
        # the surface brightness must be rescaled to be in good units
        return self.surface_brightness(r, nh[..., None]) * self.r_500 + 10 ** log_bkg


class XraySurfaceBrightnessBetaModel(hk.Module):
    """
    Xray surface brightness from a 2D surface brightness Beta-model
    """

    def __call__(self, r):
        log_bkg = hk.get_parameter("log_bkg", [], init=Constant(-5.))
        log_e_0 = hk.get_parameter("log_e_0", [], init=Constant(-4))
        log_r_c = hk.get_parameter("log_r_c", [], init=Constant(-1.))
        beta = hk.get_parameter("beta", [], init=Constant(2 / 3))

        e_0 = 10**log_e_0
        r_c = 10**log_r_c

        u_beta = jnp.sqrt(jnp.pi)*jnp.exp(gammaln(3*beta - 1/2)-gammaln(3*beta))

        return u_beta*e_0*(1+(r/r_c)**2)**(1/2-3*beta) + 10 ** log_bkg
