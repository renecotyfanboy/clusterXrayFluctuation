import jax
import jax.numpy as jnp
import haiku as hk
from jax.random import split
from ..simulation.mock_power_spectrum import MockPowerSpectrum, MockPowerSpectrumRatio
from ..utils.misc import rng_key


def chexmate_low_scale(z):
    
    slope = 0.12345876685435539
    intercept = 0.01443680039884723
    offset = 0.008775859142608464
    
    return z*slope + intercept + offset


class SpectraGenerator:
    
    def __init__(self, data, mean_model, mask, scales, ratio=False):
        
        self.data = data
        self.mean_model = mean_model
        self.scales = scales 
        self.batch = 40
        
        if not ratio:
        
            self.ps = hk.transform(lambda: MockPowerSpectrum(data, mask=mask)(scales))
        
        else: 
            
            self.ps = hk.transform(lambda: MockPowerSpectrumRatio(data, mask=mask)(scales))
        
        self.ref_params = self.ps.init(rng_key())
        self.mapped_ps = jax.pmap(self.ps.apply)

    def gen_parameters(self, ps_params, size):

        new_params = self.ref_params.copy()
        mean_model_params = self.mean_model.sample(size)
        params_values = {**mean_model_params, **ps_params}

        for sub_dict in new_params.values():

            for key, values in sub_dict.items():
                sub_dict[key] = params_values[key]

        return new_params
        
    def __call__(self, log_c, log_inj, alpha):
        
        n_spectra = len(log_c)
        ps_list = []
        
        for i in range(0, n_spectra, self.batch):

            ps_params = {'log_c': log_c[i:i+self.batch],
                         'log_inj': log_inj[i:i+self.batch], 
                         'alpha': alpha[i:i+self.batch]}

            pars = self.gen_parameters(ps_params, len(ps_params['log_c']))
            key = split(rng_key(), len(ps_params['log_c']))

            ps_list.append(self.mapped_ps(pars, key))
            
        return jnp.concatenate(ps_list)
