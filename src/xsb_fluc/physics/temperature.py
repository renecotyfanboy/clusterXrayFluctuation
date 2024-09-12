""" Python module containing models for the temperature profile of the ICM."""
import haiku as hk
import jax.numpy as jnp


class GhirardiniModel(hk.Module):
    """
    Universal temperature profile as defined in Ghirardini 2018+ in the X-COP cluster sample
    """

    def __init__(self):
        super(GhirardiniModel, self).__init__()

    def __call__(self, r):
        """
        Compute the temperature function for a given radius.

        Parameters:
            r (jnp.array): Radius to compute the temperature in $R_{500}$ units

        Returns:
            (jnp.array): Temperature function evaluated at the given radius in keV
        """

        T0 = 1.21
        rcool = jnp.exp(-2.78)
        rt = 0.34
        TmT0 = 0.5
        acool = 1.03
        c2 = 0.27

        T500 = 7.  # keV

        term1 = (TmT0 + (r / rcool) ** acool)
        term2 = (1 + (r / rcool) ** acool) * (1 + (r / rt) ** 2) ** c2

        return T500 * T0 * term1 / term2
