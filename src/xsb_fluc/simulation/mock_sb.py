import haiku as hk
import jax
import jax.numpy as jnp
from typing import Tuple
from haiku.initializers import Constant
from ..data.cluster import Cluster
from .cube import EmissivityCube, FluctuationCube


class MockSurfaceBrightness(hk.Module):
    """
    Compute mock surface brightness fluctuations
    """

    def __init__(self, data: Cluster):
        """
        Constructor for MockSurfaceBrightness using a Cluster object
        """

        super(MockSurfaceBrightness, self).__init__()
        self.pixel_size = data.kpc_per_pixel.value
        self.emissivity = EmissivityCube(data)
        self.fluctuation = FluctuationCube(data)
        self.exp = data.exp
        self.bkg = data.bkg

    def __call__(self) -> Tuple[jax.Array, jax.Array]:
        """
        Compute an X-ray surface brightness image and return the surface brightness
        image with perturbations and the unperturbed image
        """

        eps = self.emissivity()
        delta = self.fluctuation()
        log_bkg = hk.get_parameter("log_bkg", [], init=Constant(-5.))
        surface_brightness_rest = jnp.trapz(eps, dx=self.pixel_size) + 10**log_bkg
        surface_brightness_perturbed = jnp.trapz(eps * (1 + delta) ** 2, dx=self.pixel_size) + 10**log_bkg

        return surface_brightness_perturbed, surface_brightness_rest
