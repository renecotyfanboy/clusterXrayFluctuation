import jax
import jax.numpy as jnp
import haiku as hk
from jax.random import split
from ..simulation.mock_fluctuation import MockFluctuationImage
from ..utils.misc import rng_key, set_params

class ImageGenerator:

    def __init__(self, data, mean_model, freeze=False, ret_cube=False):

        self.data = data
        self.mean_model = mean_model
        self.scales = jnp.geomspace(0.05, 1., 20)
        self.batch = 40

        self.img = hk.transform(lambda: MockFluctuationImage(data)(scales=self.scales, ret_cube=ret_cube))
        self.freeze = freeze
        self.ref_params = self.img.init(rng_key())
        self.mapped_img = jax.pmap(self.img.apply)
        self.ret_cube = ret_cube

    def gen_parameters(self, ps_params, size):

        mean_model_params = self.mean_model.sample(size, freeze=self.freeze)
        params_values = {**mean_model_params, **ps_params}

        return set_params(self.ref_params, params_values), params_values

    def __call__(self, log_sigma, log_inj, alpha):


        ps_params = {'log_sigma': log_sigma,
                     'log_inj': log_inj,
                     'alpha': alpha}

        pars, pars_val = self.gen_parameters(ps_params, len(ps_params['log_sigma']))
        key = split(rng_key(), len(ps_params['log_sigma']))

        if not self.ret_cube:

            counts_p, counts_r = self.mapped_img(pars, key)

            return counts_p, counts_r, pars_val

        else:

            counts_p, counts_r, delta = self.mapped_img(pars, key)

            return counts_p, counts_r, pars_val, delta