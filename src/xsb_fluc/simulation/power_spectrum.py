import haiku as hk
import jax.numpy as jnp
from .mexican_hat import MexicanHat


class PowerSpectrum(hk.Module):
    """
    Compute the power spectrum using the Mexican Hat filter for `Cluster`data.
    """
    
    def __init__(self, data, mask=None):
        """
        Constructor for the PowerSpectrum class.

        Parameters:
            data (Cluster): Cluster object containing the data.
            mask (np.ndarray): Mask to apply to the data.
        """
        super(PowerSpectrum, self).__init__()
        self.mexican_hat = MexicanHat(data, mask=mask)
        
    def __call__(self, image, scales):
        """
        Computes the power spectrum for a given image and scales.

        Parameters:
            image (np.ndarray): Image to compute the power spectrum of.
            scales (np.ndarray): Scales to compute the power spectrum at.
        """

        return hk.vmap(lambda s: self.mexican_hat(image, s), split_rng=False)(scales)


class LogDerivativePowerSpectrum(hk.Module):

    def __init__(self, data, mask=None):
        super(LogDerivativePowerSpectrum, self).__init__()
        self.mexican_hat = MexicanHat(data, mask=mask)

    def __call__(self, image, scales):

        log_scale = jnp.log(scales)

        return hk.vmap(hk.grad(lambda log_s: jnp.log(self.mexican_hat(image, jnp.exp(log_s)))), split_rng=False)(log_scale)