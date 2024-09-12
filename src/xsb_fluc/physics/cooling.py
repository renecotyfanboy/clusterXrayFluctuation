""" Python module containing structures to compute the cooling function approximation for the ICM."""
import numpy as np
import haiku as hk
import astropy.units as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from astropy.cosmology import FlatLambdaCDM
from jax.scipy.optimize import minimize
from ..utils.config import config


class CoolingFunctionGrid:

    def __init__(self, redshift=0.01, n_points=20, kT_span=(1., 10.), nH_span=(9e19, 2e21)):
        """
        Awfully coded class to compute the cooling function grid for a given redshift.
        The grid is saved in a .npy file in the results folder. XSPEC is not needed if the
        grid is already computed. Else it will be computed using XSPEC.

        """

        self.redshift = round(redshift, 5)
        self.n_points = n_points
        self.kT_span = kT_span
        self.nH_span = nH_span

        pars = (self.redshift, n_points, *kT_span, *nH_span)

        self.kT, self.nH = np.meshgrid(np.linspace(*kT_span, n_points),
                                       np.geomspace(*nH_span, n_points),
                                       indexing='ij')

        cooling_path = os.path.join(config.RESULTS_PATH, 'cooling_database', f'{pars}.npy')

        if not os.path.exists(cooling_path):

            print('Cooling grid not found! Computing with XSPEC')

            from .countrate import PHabsAPEC

            phabs = PHabsAPEC()

            abundance = 0.3
            norm = 1.
            countrate = phabs.countrate(self.nH / 1e22, self.kT, abundance, float(self.redshift), norm)

            # XSPEC norm factor for an apec model
            dc = FlatLambdaCDM(70, 0.3).comoving_distance(redshift).to(u.Mpc)
            factor = 1e-14 / (4 * np.pi * dc ** 2) / 1.17

            # Units of the countrate over units of the norm of APEC model
            cooling = factor * countrate * (u.count * u.s ** (-1) * u.cm ** 5)

            # Conversion to the best unit for Lambda * ne^2
            self.cooling = (cooling.to(u.count * u.cm ** 6 / u.kpc ** 3 / u.s)).value
            np.save(cooling_path, self.cooling)

        else:

            self.cooling = np.load(cooling_path)


class CoolingFunctionModel(hk.Module):
    r"""
    Cooling function model using a grid of precomputed cooling function. It is fitted to the grid using
    a least square method on the fly. The model is given by the following formula:

    $$\Psi(N_H, T) \simeq \Lambda_0 e^{- N_H \sigma} \left( \frac{T}{T_{\text{break}}}\right)^{-\alpha_{1}}\left(\frac{1}{2} + \frac{1}{2}\left(\frac{T}{T_{\text{break}}}\right)^{1/\Delta}\right)^{(\alpha_1 - \alpha_2)\Delta}$$

    !!! note
        $\Psi(N_H, T)$ here is different from $\Lambda(T)$ as it includes the instrumental convolution
    """

    def __init__(self, coolingFunctionGrid):
        super(CoolingFunctionModel, self).__init__()
        self.grid = coolingFunctionGrid
        # Norm, sigma, kTbreak, alpha1, alpha2, delta
        X = jnp.array([0.01, 3., 1.65, 1.7, 0.14, 0.13])

        self.res = minimize(self.fitness, X, method="BFGS", tol=1e-15)
        self.pars = self.res.x

    def kT_dependency(self, kT, kT_break, alpha1, alpha2, delta):
        t = (kT / kT_break)

        return t ** (-alpha1) * (1 / 2 * (1 + t ** (1 / delta))) ** ((alpha1 - alpha2) * delta)

    def nH_dependency(self, nH, sigma):
        return jnp.exp(-nH / 1e22 * sigma)

    def model(self, nH, kT, pars):
        return pars[0] * self.nH_dependency(nH, pars[1]) * self.kT_dependency(kT, *pars[2:])

    def __call__(self, nH, kT):
        return self.model(nH, kT, self.pars)

    def fitness(self, pars):
        lsq = (self.grid.cooling - self.model(self.grid.nH, self.grid.kT, pars)) ** 2

        return jnp.sum(lsq / self.grid.cooling ** 2) ** 1 / 2

    @property
    def residual(self):
        lsq = (self.grid.cooling - self.model(self.grid.nH, self.grid.kT, self.pars)) ** 2

        return lsq ** (1 / 2) / self.grid.cooling

    def plot_residual(self):
        plt.figure()
        plt.contourf(self.grid.kT, self.grid.nH, self.residual)
        plt.colorbar()
