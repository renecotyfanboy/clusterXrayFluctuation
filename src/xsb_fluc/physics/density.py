""" Python module containing models for the density profile of the ICM."""

import haiku as hk
import jax.numpy as jnp
from haiku.initializers import Constant


class CleanVikhlininModel(hk.Module):
    r"""
    Density model which use a modified Vikhlinin functional form, with alpha fixed to 0 and gamma to 3

    $$n_{e}^2(r) = n_{e,0}^2 \left( 1 + \left( \frac{r}{r_{c}} \right)^{2} \right)^{-3\beta} \left(1+\left( \frac{r}{r_{c}} \right)^{\gamma} \right)^{-\frac{\epsilon}{\gamma}}$$
    """

    def __init__(self):
        super(CleanVikhlininModel, self).__init__()

    def __call__(self, r: jnp.array) -> jnp.array:
        """Compute the density function for a given radius.

        Parameters:
            r (jnp.array): Radius to compute the density function in R500 units

        Returns:
            (jnp.array): Density function evaluated at the given radius in cm$^{-6}$
        """
        log_ne2 = hk.get_parameter("log_ne2", [], init=Constant(-3.))
        log_r_c = hk.get_parameter("log_r_c", [], init=Constant(-1.))
        log_r_s = hk.get_parameter("log_r_s", [], init=Constant(-0.1))
        beta = hk.get_parameter("beta", [], init=Constant(0.6))
        epsilon = hk.get_parameter("epsilon", [], init=Constant(3.))

        gamma = 3.

        ne2 = 10 ** log_ne2
        r_c = 10 ** log_r_c
        r_s = 10 ** log_r_s

        term1 = (1. + (r / r_c) ** 2) ** (-3 * beta)
        term2 = (1. + (r / r_s) ** gamma) ** (-epsilon / gamma)

        return ne2 * term1 * term2


class BetaModel(hk.Module):
    r"""Density model which use a beta-model formula

    $$n_{e}^2(r) = n_{e,0}^2 \left( 1 + \left( \frac{r}{r_{c}} \right)^{2} \right)^{-3\beta}$$
    """
    def __init__(self):
        super(BetaModel, self).__init__()

    def __call__(self, r):
        """Compute the density function for a given radius.

        Parameters:
            r (jnp.array): Radius to compute the density function in R500 units

        Returns:
            (jnp.array): Density function evaluated at the given radius in cm$^{-6}$
        """
        log_ne2 = hk.get_parameter("log_ne2", [], init=Constant(-3.))
        log_r_c = hk.get_parameter("log_r_c", [], init=Constant(-1))
        beta = hk.get_parameter("beta", [], init=Constant(2/3))

        ne2 = 10 ** log_ne2
        r_c = 10 ** log_r_c

        term1 = (1. + (r / r_c) ** 2) ** (-3 * beta)

        return ne2 * term1
