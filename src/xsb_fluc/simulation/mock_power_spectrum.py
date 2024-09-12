import haiku as hk
import jax.numpy as jnp
from xsb_fluc.simulation.power_spectrum import PowerSpectrum
from ..data.cluster import Cluster
from .mock_fluctuation import MockFluctuationImage


class MockPowerSpectrum(hk.Module):
    """
    Compute X-ray surface brightness power spectra.
    """
    def __init__(self, data: Cluster, mask=None):
        super(MockPowerSpectrum, self).__init__()
        self.power_spectrum = PowerSpectrum(data, mask=mask)
        self.mock_image = MockFluctuationImage(data)
        self.exp = data.exp
        self.bkg = data.bkg

    def __call__(self, scales):

        counts_perturbed, counts = self.mock_image()
        fluctuation = jnp.where(self.exp > 0, (counts_perturbed - counts) / (2*self.exp), 0.)

        return self.power_spectrum(fluctuation, scales)

    
class MockPowerSpectrumRatio(hk.Module):

    def __init__(self, data: Cluster, mask=None):
        super(MockPowerSpectrumRatio, self).__init__()
        self.power_spectrum = PowerSpectrum(data, mask=mask)
        self.mock_image = MockFluctuationImage(data)
        self.exp = data.exp
        self.bkg = data.bkg

    def __call__(self, scales):

        counts_perturbed, counts = self.mock_image()
        fluctuation = jnp.where(self.exp > 0, counts_perturbed/counts, 0.)

        return self.power_spectrum(fluctuation, scales)