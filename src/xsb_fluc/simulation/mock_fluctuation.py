import haiku as hk
import jax
import jax.numpy as jnp
from typing import Tuple, Union, Any
from jax import random, Array
from haiku.initializers import Constant
from numpy import ndarray, dtype, floating

from ..data.cluster import Cluster
from .cube import EmissivityCube, FluctuationCube


class MockFluctuationImage(hk.Module):
    """
    Compute an X-ray surface brightness image including density fluctuations.
    """

    def __init__(self, data: Cluster):
        """
        Constructor for MockFluctuationImage using a Cluster object
        """

        super(MockFluctuationImage, self).__init__()
        self.pixel_size = data.kpc_per_pixel.value
        self.emissivity = EmissivityCube(data)
        self.fluctuation = FluctuationCube(data)
        self.exp = data.exp
        self.bkg = data.bkg

    def __call__(self, ret_cube=False) -> Union[
        tuple[Array, Array, Array], tuple[Array, Array]]:
        """
        Compute an X-ray surface brightness image and return either the count image
        with perturbations and the unperturbed image, or the cubes with the two previous.

        Parameters:
            ret_cube (bool): Whether to return the cube or not.
        """

        eps = self.emissivity()
        delta = self.fluctuation()
        log_bkg = hk.get_parameter("log_bkg", [], init=Constant(-5.))
        surface_brightness_rest = jnp.trapz(eps, dx=self.pixel_size) + 10**log_bkg
        surface_brightness_perturbed = jnp.trapz(eps * (1 + delta) ** 2, dx=self.pixel_size) + 10**log_bkg
        counts_perturbed = random.poisson(hk.next_rng_key(), surface_brightness_perturbed*self.exp + self.bkg)
        counts = surface_brightness_rest*self.exp + self.bkg

        if ret_cube:
            return counts_perturbed, counts, delta

        return counts_perturbed, counts
