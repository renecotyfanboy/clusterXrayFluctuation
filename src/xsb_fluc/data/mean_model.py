import os
import arviz as az
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from ..simulation.mock_image import MockXrayCounts, MockXrayCountsUnmasked
from ..physics.ellipse import EllipseRadius, Angle
from xsb_fluc.simulation.power_spectrum import PowerSpectrum
from .cluster import Cluster
from numpy.random import default_rng
from ..utils.config import config

rng = default_rng()


def inference_to_params(values):
    
    return {'mock_xray_counts/~/ellipse_radius': 
                {'angle': jnp.asarray(values['angle']),
                  'eccentricity': jnp.asarray(values['eccentricity']),
                  'x_c': jnp.asarray(values['x_c']),
                  'y_c': jnp.asarray(values['y_c'])},
         'mock_xray_counts/~/xray_surface_brightness': 
            {'log_bkg': jnp.asarray(values['log_bkg'])},
         'mock_xray_counts/~/xray_surface_brightness/~/xray_emissivity/~/clean_vikhlinin_model': 
            {'log_ne2': jnp.asarray(values['log_ne2']),
              'log_r_c': jnp.asarray(values['log_r_c']),
              'log_r_s': jnp.asarray(values['log_r_s']),
              'beta': jnp.asarray(values['beta']),
              'epsilon': jnp.asarray(values['epsilon'])}}


def inference_to_params_unmasked(values):
    
    return {'mock_xray_counts_unmasked/~/ellipse_radius': 
                {'angle': jnp.asarray(values['angle']),
                  'eccentricity': jnp.asarray(values['eccentricity']),
                  'x_c': jnp.asarray(values['x_c']),
                  'y_c': jnp.asarray(values['y_c'])},
         'mock_xray_counts_unmasked/~/xray_surface_brightness': 
            {'log_bkg': jnp.asarray(values['log_bkg'])},
         'mock_xray_counts_unmasked/~/xray_surface_brightness/~/xray_emissivity/~/clean_vikhlinin_model': 
            {'log_ne2': jnp.asarray(values['log_ne2']),
              'log_r_c': jnp.asarray(values['log_r_c']),
              'log_r_s': jnp.asarray(values['log_r_s']),
              'beta': jnp.asarray(values['beta']),
              'epsilon': jnp.asarray(values['epsilon'])}}

class MeanModel:
    """
    Awfully coded class to handle the results of the mean model fitting.

    Attributes:
        data: the cluster data.
        mean_model: the mean model function.
        radius: the radius function.
        angle: the angle function.
        inference_data: the inference data.
        true_image: the true image.
        posterior_median: the posterior median.
        posterior_params: the posterior parameters.
        ellipse_params: the ellipse parameters.
        number_of_samples: the number of samples.
        best_fit: the best fit image.
    """
    
    def __init__(self, data: Cluster, inference_data: az.InferenceData, model='all'):
        """
        Constructor for the `MeanModel` class.

        Parameters:
            data: the cluster data.
            inference_data: the inference data.
            model: (artefact of code from the X-COP paper).
        """
        
        self.data = data
        self.mean_model = jax.jit(hk.without_apply_rng(hk.transform(lambda : MockXrayCounts(data)())).apply)
        self.mean_model_unmasked = jax.jit(hk.without_apply_rng(hk.transform(lambda : MockXrayCountsUnmasked(data)())).apply)
        self.radius = hk.without_apply_rng(hk.transform(lambda x, y :EllipseRadius.from_data(data)(x, y))).apply
        self.angle = hk.without_apply_rng(hk.transform(lambda x, y :Angle.from_data(data)(x, y))).apply
        self.inference_data = inference_data
        self.true_image = jnp.array(data.img, dtype=jnp.float32)
        stacked = inference_data.posterior.stack(draws=("chain", "draw"))
        var_name = ['log_ne2', 'log_r_c', 'log_r_s', 'beta', 'epsilon', 'log_bkg','angle', 'eccentricity','x_c', 'y_c']
        self.posterior_median = self.inference_data.posterior.median()
        self.posterior_params = {name:jnp.asarray(stacked[name].values) for name in var_name}
        self.ellipse_params = {'ellipse_radius': 
                            {'angle': jnp.asarray(self.posterior_median['angle']),
                            'eccentricity': jnp.array(self.posterior_median['eccentricity']),
                            'x_c': jnp.asarray(self.posterior_median['x_c']),
                            'y_c': jnp.asarray(self.posterior_median['y_c'])}}

        self.number_of_samples = 10000
        self.best_fit = self.mean_model(inference_to_params(self.posterior_median))
        self.model_c = model

    @classmethod
    def from_data(cls, data: Cluster):
        """
        Load the mean model associated to a given cluster if it exists.
        """

        name = data.name
        posterior = az.from_netcdf(os.path.join(config.RESULTS_PATH, f'mean_model/{name}_all.posterior'))
        return cls(data, posterior)
    
    @property
    def best_fit_unmasked(self):
        return self.mean_model_unmasked(inference_to_params_unmasked(self.posterior_median))
    
    @property
    def rad(self):
        """
        Return best-fit radius for each pixel.
        """
        median = self.posterior_median
        ellipse_params = {'ellipse_radius': 
                            {'angle': jnp.asarray(median['angle']),
                            'eccentricity': jnp.array(median['eccentricity']),
                            'x_c': jnp.asarray(median['x_c']),
                            'y_c': jnp.asarray(median['y_c'])}}

        return self.radius(ellipse_params, self.data.x_ref, self.data.y_ref)


    @property
    def ang(self):
        """
        Return best-fit angle for each pixel.
        """
        median = self.posterior_median
        angle_params = {'angle':
                            {'x_c': jnp.asarray(median['x_c']),
                             'y_c': jnp.asarray(median['y_c']),
                             'angle': jnp.asarray(median['angle'])}}


        return self.angle(angle_params, self.data.x_ref, self.data.y_ref)

    @property
    def angle_sample(self):
        """
        Return a sample of angles for each pixel.
        """
        angle_params = {'angle':
                            {'x_c': self.posterior_params['x_c'],
                             'y_c': self.posterior_params['y_c'],
                             'angle': self.posterior_params['angle']}}

        def func(pars):
            return self.angle(pars, self.data.x_ref, self.data.y_ref)

        return jax.vmap(func)(angle_params)

    @property
    def rad_sample(self):
        """
        Return a sample of radii for each pixel.
        """
        ellipse_params = {'ellipse_radius':
                              {'angle': self.posterior_params['angle'],
                               'eccentricity': self.posterior_params['eccentricity'],
                               'x_c': self.posterior_params['x_c'],
                               'y_c': self.posterior_params['y_c']}}

        def func(pars):
            return self.radius(pars, self.data.x_ref, self.data.y_ref)

        return jax.vmap(func)(ellipse_params)


    @property
    def rad_circ_sample(self):

        ellipse_params = {'ellipse_radius':
                              {'angle': 0*self.posterior_params['angle'],
                               'eccentricity': 0.*self.posterior_params['eccentricity'],
                               'x_c': self.posterior_params['x_c'],
                               'y_c': self.posterior_params['y_c']}}

        def func(pars):
            return self.radius(pars, self.data.x_ref, self.data.y_ref)

        return jax.vmap(func)(ellipse_params)


    def compute_rad(self, x, y):
        median = self.posterior_median
        ellipse_params = {'ellipse_radius': 
                            {'angle': jnp.asarray(median['angle']),
                            'eccentricity': jnp.array(median['eccentricity']),
                            'x_c': jnp.asarray(median['x_c']),
                            'y_c': jnp.asarray(median['y_c'])}}

        return self.radius(ellipse_params, x, y)
    
    @property
    def fluctuation_absolute(self):
        """
        Return the absolute fluctuation map.
        """
        exp = jnp.asarray(self.data.exp)
        img = jnp.asarray(self.true_image)
        fit = jnp.asarray(self.best_fit)
        
        return jnp.where(exp>0., (img - fit)/(2*exp), 0.)

    @property
    def fluctuation_relative(self):
        """
        Return the relative fluctuation map.
        """
        return jnp.where(self.data.exp > 0., jnp.nan_to_num(jnp.abs(self.true_image)/jnp.abs(self.best_fit)), 0.)

    def power_spectrum_absolute(self, scales=np.geomspace(0.05, 1., 20),mask=None):
        """
        Compute the absolute power spectrum with a given mask.

        Parameters:
            scales: array of scales to compute the power spectrum.
            mask: mask to apply to the data.

        !!! warning
            This function might not work ?
        """

        power_spectrum = hk.without_apply_rng(hk.transform(lambda img: PowerSpectrum(self.data, mask=mask)(img, scales)))
        
        return power_spectrum.apply(None, self.fluctuation_absolute)
    
    def power_spectrum_relative(self, mask=None):
        """
        Compute the relative power spectrum with a given mask.

        Parameters:
            scales: array of scales to compute the power spectrum.
            mask: mask to apply to the data.

        !!! warning
            This function might not work ?
        """
        scales = np.geomspace(0.05, 1., 20)
        power_spectrum = hk.without_apply_rng(hk.transform(lambda img: PowerSpectrum(self.data, mask=mask)(img, scales)))
        
        return power_spectrum.apply(None, self.fluctuation_relative)
    
    def sample(self, size, freeze=False):
        
        if not freeze:

            samples = rng.choice(self.number_of_samples, size=size, replace=False)
            return {key: value[samples] for key, value in self.posterior_params.items()}

        else:

            return {key: jnp.median(value)*jnp.ones((size,)) for key, value in self.posterior_params.items()}
