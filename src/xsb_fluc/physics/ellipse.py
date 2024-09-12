""" Python module containing models for the ellipticity of projected surface brightness."""

import haiku as hk
import astropy.units as u
import jax.numpy as jnp
from haiku.initializers import Constant
from ..data.cluster import Cluster


class EllipseRadius(hk.Module):
    """
    Include ellipticity in 2D radius maps
    """

    def __init__(self, x_c, y_c, pixel_size):
        super(EllipseRadius, self).__init__()
        self.x_c_init = x_c
        self.y_c_init = y_c
        self.pixel_size = pixel_size

    @classmethod
    def from_data(cls, data: "Cluster"):
        """
        Build the ellipticity model from a `Cluster` object.
        """

        pixel_size = (data.kpc_per_pixel/data.r_500).to(1 / u.pix).value

        return cls(data.x_c, data.y_c, pixel_size)

    def __call__(self, x_ref, y_ref):
        """
        Compute the elliptical radius for a given position.

        Returns:
            (jnp.array): Elliptical radius in unit of $R_{500}$
        """
        angle = hk.get_parameter("angle", [], init=Constant(0.))
        e = hk.get_parameter("eccentricity", [], init=Constant(0.))
        x_c = self.x_c_init*(1+hk.get_parameter("x_c", [], init=Constant(0.)))
        y_c = self.y_c_init*(1+hk.get_parameter("y_c", [], init=Constant(0.)))

        x_tilde = (x_ref-x_c)*jnp.cos(angle) - (y_ref-y_c)*jnp.sin(angle)
        y_tilde = (y_ref-y_c)*jnp.cos(angle) + (x_ref-x_c)*jnp.sin(angle)
        r = jnp.sqrt(x_tilde**2*(1-e**2)**(-1/2) + y_tilde**2*(1-e**2)**(1/2))*self.pixel_size

        return r


class Angle(hk.Module):

    def __init__(self, x_c, y_c, pixel_size):
        super(Angle, self).__init__()
        self.x_c_init = x_c
        self.y_c_init = y_c
        self.pixel_size = pixel_size

    @classmethod
    def from_data(cls, data: "Cluster"):
        pixel_size = (data.kpc_per_pixel/data.r_500).to(1 / u.pix).value

        return cls(data.x_c, data.y_c, pixel_size)

    def __call__(self, x_ref, y_ref):

        x_c = self.x_c_init*(1+hk.get_parameter("x_c", [], init=Constant(0.)))
        y_c = self.y_c_init*(1+hk.get_parameter("y_c", [], init=Constant(0.)))

        return (jnp.arctan2((x_ref-x_c),(y_ref-y_c)) + hk.get_parameter("angle", [], init=Constant(0.)) + jnp.pi/2)%(2*jnp.pi)