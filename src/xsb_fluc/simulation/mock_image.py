import haiku as hk
import jax.numpy as jnp
from src.xsb_fluc.physics.surface_brightness import XraySurfaceBrightness, XraySurfaceBrightnessBetaModel
from src.xsb_fluc.physics.ellipse import EllipseRadius
from ..data.cluster import Cluster


class MockXrayCounts(hk.Module):

    def __init__(self, data: Cluster):

        super(MockXrayCounts, self).__init__()
        self.surface_brightness = XraySurfaceBrightness.from_data(data)
        self.radius = EllipseRadius.from_data(data)

        self.bkg = jnp.asarray(data.bkg, dtype=jnp.float32)
        self.exp = jnp.asarray(data.exp, dtype=jnp.float32)
        self.x_ref = jnp.asarray(data.x_ref, dtype=jnp.float32)
        self.y_ref = jnp.asarray(data.y_ref, dtype=jnp.float32)
        self.nH = jnp.asarray(data.nh, dtype=jnp.float32)

    def __call__(self):

        r = self.radius(self.x_ref, self.y_ref)
        surface_brightness = self.surface_brightness(r, self.nH)

        return surface_brightness * self.exp + self.bkg


class MockXrayCountsBetaModel(hk.Module):

    def __init__(self, data: Cluster):

        super(MockXrayCountsBetaModel, self).__init__()
        self.surface_brightness = XraySurfaceBrightnessBetaModel()#.from_data(data)
        self.radius = EllipseRadius.from_data(data)

        self.bkg = jnp.asarray(data.bkg, dtype=jnp.float32)
        self.exp = jnp.asarray(data.exp, dtype=jnp.float32)
        self.x_ref = jnp.asarray(data.x_ref, dtype=jnp.float32)
        self.y_ref = jnp.asarray(data.y_ref, dtype=jnp.float32)
        self.nH = jnp.asarray(data.nh, dtype=jnp.float32)

    def __call__(self):

        r = self.radius(self.x_ref, self.y_ref)
        surface_brightness = self.surface_brightness(r)

        return surface_brightness * self.exp + self.bkg


class MockXrayCountsUnmasked(hk.Module):

    def __init__(self, data: Cluster):

        super(MockXrayCountsUnmasked, self).__init__()
        self.surface_brightness = XraySurfaceBrightness.from_data(data)
        self.radius = EllipseRadius.from_data(data)

        self.bkg = jnp.asarray(data.bkg, dtype=jnp.float32)
        self.exp = jnp.asarray(data.unmasked_exposure, dtype=jnp.float32)
        self.x_ref = jnp.asarray(data.x_ref, dtype=jnp.float32)
        self.y_ref = jnp.asarray(data.y_ref, dtype=jnp.float32)
        self.nH = jnp.asarray(data.nh, dtype=jnp.float32)

    def __call__(self):

        r = self.radius(self.x_ref, self.y_ref)
        surface_brightness = self.surface_brightness(r, self.nH)

        return surface_brightness * self.exp + self.bkg
