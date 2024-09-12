import numpy as np
import jax.numpy as jnp
import jax.numpy.fft as fft
import astropy.units as u
from ..data.cluster import Cluster

class SpatialGrid3D:
    """
    Helper function to define a spatial grid to simulate a 3D cluster
    """

    def __init__(self, pixsize=2./1000., shape=(100, 100), crop_r_500=5.):
        r"""
        Constructor for SpatialGrid3D.

        Parameters:
            pixsize (float): Size of the pixel in $R_{500}$ units.
            shape (tuple): Shape of the cube on the sky.
            crop_r_500 (float): Size along the line of sight in $\pm R_{500}$.
        """

        self.pixsize = pixsize

        x_size, y_size = shape
        z_size = np.ceil(2*crop_r_500/pixsize).astype(int)

        self.x = jnp.linspace(-self.pixsize * (x_size - 1)/2, self.pixsize * (x_size - 1)/2, x_size)
        self.y = jnp.linspace(-self.pixsize * (y_size - 1)/2, self.pixsize * (y_size - 1)/2, y_size)
        self.z = jnp.linspace(-self.pixsize * (z_size - 1)/2, self.pixsize * (z_size - 1)/2, z_size)
        self.X, self.Y, self.Z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij', sparse=True)
        self.y_i, self.x_i = jnp.indices(shape)
        self.R = jnp.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.shape = (x_size, y_size, z_size)
        self.volume = np.prod(self.shape)*pixsize**3

    @classmethod
    def from_data(cls, data: Cluster, crop_r_500=5., pixsize=None):
        """
        Constructor for SpatialGrid3D using a Cluster object

        Parameters:
            crop_r_500: size along the line of sight.
            pixsize: should be None since it is read from the Cluster
        """
        
        if pixsize is None:
        
            return cls(pixsize=(data.kpc_per_pixel/data.r_500).to(1/u.pixel).value,
                       shape = data.shape, 
                       crop_r_500=crop_r_500)
        
        else : 
            
            return cls(pixsize=pixsize,
                       shape = data.shape, 
                       crop_r_500=crop_r_500)
            

class FourierGrid3D:
    """
    The equivalent of a Spatial grid in Fourier space
    """

    def __init__(self, spatial_grid: SpatialGrid3D):
        """
        Constructor of a FourierGrid3D object as the dual of a SpatialGrid3D
        """

        self.kx = fft.fftfreq(len(spatial_grid.x), d=spatial_grid.x[1] - spatial_grid.x[0])
        self.ky = fft.fftfreq(len(spatial_grid.y), d=spatial_grid.y[1] - spatial_grid.y[0])
        self.kz = fft.rfftfreq(len(spatial_grid.z), d=spatial_grid.z[1] - spatial_grid.z[0])
        KX, KY, KZ = jnp.meshgrid(self.kx, self.ky, self.kz, indexing='ij', sparse=True)
        self.K = jnp.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

        self.shape = self.K.shape