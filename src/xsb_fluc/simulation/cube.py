import haiku as hk
import jax.typing
import numpy as np
import jax.numpy as jnp
import jax.random as random
import jax.numpy.fft as fft
from ..data.cluster import Cluster
from ..physics.emissivity import XrayEmissivity
from ..physics.ellipse import EllipseRadius
from ..simulation.grid import SpatialGrid3D, FourierGrid3D
from ..physics.turbulence import KolmogorovPowerSpectrum


class EmissivityCube(hk.Module):
    """
    Compute an emissivity cube for a given cluster. Used as a part of simulations.
    """

    def __init__(self, data: Cluster):
        """
        Constructor for EmissivityCube using a Cluster object
        """

        super(EmissivityCube, self).__init__()
        self.emissivity = XrayEmissivity.from_data(data)
        self.radius = EllipseRadius.from_data(data)
        self.spatial_grid = SpatialGrid3D.from_data(data, crop_r_500=5)
        self.nh = data.nh
        self.x_i, self.x_j = self.spatial_grid.x_i, self.spatial_grid.y_i
        self.Z = self.spatial_grid.Z
        
    def __call__(self) -> jax.Array:
        """
        Return the emissivity cube.
        """
        
        Rho = self.radius(self.x_i, self.x_j) 
        R = jnp.sqrt(Rho[..., None]**2 + self.Z**2)
        epsilon = self.emissivity(R, self.nh[..., None])
        
        return epsilon


class FluctuationCube(hk.Module):
    """
    Compute a fluctuation cube for a given cluster. The density fluctuations are computed
    assuming a Gaussian Random Field defined with a turbulent power spectrum.
    """

    def __init__(self, data: Cluster):
        """
        Constructor for FluctuationCube using a Cluster object
        """

        super(FluctuationCube, self).__init__()
        self.power_spectrum = KolmogorovPowerSpectrum()
        self.spatial_grid = SpatialGrid3D.from_data(data, crop_r_500=5)
        self.fourier_grid = FourierGrid3D(self.spatial_grid)
        self.K = self.fourier_grid.K.astype(np.float32)
        
    def __call__(self) -> jax.Array:
        """
        Return the fluctuation cube.
        """
        
        key = hk.next_rng_key()

        field_spatial = random.normal(key, shape=self.spatial_grid.shape)
        field_fourier = fft.rfftn(field_spatial)*jnp.sqrt(self.power_spectrum(self.K)/self.spatial_grid.pixsize**3)
        field_spatial = fft.irfftn(field_fourier, s=self.spatial_grid.shape)
        
        return field_spatial
