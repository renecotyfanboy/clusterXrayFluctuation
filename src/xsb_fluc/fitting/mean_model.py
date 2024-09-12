import jax.numpy as jnp
import arviz as az
import haiku as hk
import numpyro
import numpy as np
import jax.random as random
import numpyro.distributions as dist
from typing import Literal
from numpyro.infer import MCMC, init_to_value, Predictive, BarkerMH
from src.xsb_fluc.data.cluster import Cluster
from src.xsb_fluc.simulation.mock_image import MockXrayCounts, MockXrayCountsBetaModel
from src.xsb_fluc.utils.misc import set_params, rng_key


class MeanModelFitter:
    """
    This class is meant to fit a 3D model using the X-ray surface
    """

    def __init__(self,
                 cluster: Cluster,
                 n_samples: int = 1000,
                 n_warmup: int = 1000,
                 n_chains: int = 1,
                 model: Literal[None, 'beta', 'circ'] = None,
                 max_tree_depth: int = 10,
                 ref_params=None):

        """
        Constructor for the MeanModelFitter which fit the mean model using MCMC

        Parameters:
            cluster (Cluster): the cluster object which will be fitted
            n_samples (int): the number of samples
            n_warmup (int): the number of warmup samples
            n_chains (int): the number of chains
            model (Literal[None, 'beta', 'circ']): the model to be fitted. Leave None to use the best.
            max_tree_depth (int): the maximum recursion depth of the MCMC. Lower to 7 or 5 if too slow.
            ref_params (dict): the parameters where the chain should be started.
        """

        self.cluster = cluster
        self.model_var = model
        self.ref_params = ref_params
        self.max_tree_depth = max_tree_depth

        if self.model_var == 'beta' or self.model_var == 'circ':

            self.counts = hk.without_apply_rng(hk.transform(lambda: MockXrayCountsBetaModel(self.cluster)()))

        else:

            self.counts = hk.without_apply_rng(hk.transform(lambda: MockXrayCounts(self.cluster)()))

        self.mcmc_config = {'num_samples': n_samples,
                            'num_warmup': n_warmup,
                            'num_chains': n_chains,
                            'progress_bar': True}

    @property
    def prior(self):
        """
        Default priors distributions.
        """

        sb = self.cluster.img/self.cluster.exp
        X = np.stack([self.cluster.x_ref, self.cluster.y_ref])
        np.average(X, weights=sb, axis=1)
        X_cov = np.cov(X, aweights=sb)

        eig_val, eig_vec = np.linalg.eig(X_cov)
        angle = np.arccos(np.dot([1., 0.], eig_vec[:, np.argmin(eig_val)]))
        (1-(min(eig_val)/max(eig_val))**2)**(1/2)

        if self.model_var is None:

            prior = {'angle': numpyro.sample('angle', dist.Uniform(-jnp.pi/2, jnp.pi/2)),
                     'eccentricity': numpyro.sample('eccentricity', dist.Uniform(0., 0.99)),
                     'x_c': numpyro.sample('x_c', dist.Normal(0., 0.5)),
                     'y_c': numpyro.sample('y_c', dist.Normal(0., 0.5)),
                     'log_bkg': numpyro.sample('log_bkg', dist.Uniform(-10., -4.)),
                     'log_ne2': numpyro.sample('log_ne2', dist.Uniform(-8., -3.)),
                     'log_r_c': numpyro.sample('log_r_c', dist.Uniform(-2., 0.)),
                     'log_r_s': numpyro.sample('log_r_s', dist.Uniform(-1., 1.)),
                     'beta': numpyro.sample('beta', dist.Uniform(0., 5.)),
                     'epsilon': numpyro.sample('epsilon', dist.Uniform(0., 5.))}

        if self.model_var == 'beta':

            prior = {'angle': numpyro.sample('angle', dist.TruncatedNormal(loc=angle,
                                                                           scale=0.2,
                                                                           low=-jnp.pi/2+angle,
                                                                           high=jnp.pi/2+angle)),

                     'eccentricity': numpyro.sample('eccentricity', dist.TruncatedNormal(
                                                                           loc=angle,
                                                                           scale=0.2,
                                                                           low=0,
                                                                           high=0.9)),
                     'x_c': numpyro.sample('x_c', dist.Normal(0., 0.2)),
                     'y_c': numpyro.sample('y_c', dist.Normal(0., 0.2)),
                     'log_bkg': numpyro.sample('log_bkg', dist.Uniform(-10., -4.)),
                     'log_e_0': numpyro.sample('log_e_0', dist.Uniform(-7., 0.)),
                     'log_r_c': numpyro.sample('log_r_c', dist.Uniform(-4., 1.)),
                     'beta': numpyro.sample('beta', dist.Uniform(0., 5.))}

        if self.model_var == 'circ':

            prior = {'angle': jnp.array(0.),
                     'eccentricity': jnp.array(0.),
                     'x_c': numpyro.sample('x_c', dist.Normal(0., 0.2)),
                     'y_c': numpyro.sample('y_c', dist.Normal(0., 0.2)),
                     'log_bkg': numpyro.sample('log_bkg', dist.Uniform(-10., -4.)),
                     'log_e_0': numpyro.sample('log_e_0', dist.Uniform(-7., 0.)),
                     'log_r_c': numpyro.sample('log_r_c', dist.Uniform(-4., 1.)),
                     'beta': numpyro.sample('beta', dist.Uniform(0., 5.))}

        return prior

    def model(self):
        """
        The numpyro model which is used in the fitting routine.
        """

        params = set_params(self.counts.init(None), self.prior)

        numpyro.sample('likelihood',
                       dist.Poisson(self.counts.apply(params)),
                       obs=jnp.array(self.cluster.img, dtype=jnp.float32))

    def fit(self) -> az.InferenceData:
        """
        Perform the fitting routine.

        Returns:
            An az.InferenceData object containing the fit results.
        """

        if self.ref_params is not None:

            kernel = BarkerMH(self.model,
                          #max_tree_depth=self.max_tree_depth,
                          init_strategy=init_to_value(values=self.ref_params),
                        )
                          #dense_mass=True,
                          #target_accept_prob=0.95)

        else :

            kernel = BarkerMH(self.model)
              #max_tree_depth=self.max_tree_depth,
              #dense_mass=True,
              #target_accept_prob=0.95)

        posterior = MCMC(kernel, **self.mcmc_config)

        key = random.split(rng_key(), 4)
        posterior.run(key[0])

        posterior_samples = posterior.get_samples()
        posterior_predictive = Predictive(self.model, posterior_samples)(key[1])
        prior = Predictive(self.model, num_samples=100000)(key[2])

        inference_data = az.from_numpyro(posterior, prior=prior, posterior_predictive=posterior_predictive)

        return inference_data
