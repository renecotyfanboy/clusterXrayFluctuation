import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
from astropy import uncertainty as unc
from .mean_model import MeanModel
from ..physics.surface_brightness import XraySurfaceBrightness


class RadialProfile:

    def __init__(self, mean_model: MeanModel):

        self.mean_model = mean_model
        self.data = mean_model.data
        self.rad = mean_model.rad
        self.sb = hk.without_apply_rng(hk.transform(lambda r, nh: XraySurfaceBrightness.from_data(self.data)(r, nh)))

        bin_number = 40
        max_rad = jnp.max(jnp.where(self.data.exp > 0, self.rad, 0.))
        bin_edges = jnp.linspace(0, max_rad, bin_number + 1)

        counts = np.empty((bin_number,))
        exp = np.empty((bin_number,))
        bkg = np.empty((bin_number,))
        nh = np.empty((bin_number,))
        bin_center = np.empty((bin_number,))
        bin_width = np.empty((bin_number,))

        for i in range(bin_number):
            pixels = (bin_edges[i] <= self.rad) & (self.rad < bin_edges[i + 1])
            counts[i] = jnp.sum(self.data.img[pixels])
            exp[i] = jnp.sum(self.data.exp[pixels])
            bkg[i] = jnp.sum(self.data.bkg[pixels])
            nh[i] = jnp.mean(self.data.nh[pixels])

            bin_center[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_width[i] = (bin_edges[i + 1] - bin_edges[i]) / 2

        self.counts = unc.poisson(counts, n_samples=10000)
        self.exp = exp
        self.bkg = unc.poisson(bkg, n_samples=10000)
        self.nh = nh
        self.bin_center = bin_center
        self.bin_width = bin_width

        self.surface_brightness = (self.counts - self.bkg) / self.exp
        self.e_surface_brightness = np.abs(self.surface_brightness.pdf_percentiles([16, 86]) - self.surface_brightness.pdf_median())

    def show(self):

        samples = self.mean_model.sample(1000)
        params = {'xray_surface_brightness':
                      {'log_bkg': jnp.float32(samples['log_bkg'])},
                  'xray_surface_brightness/~/xray_emissivity/~/clean_vikhlinin_model':
                      {'log_ne2': jnp.float32(samples['log_ne2']),
                       'log_r_c': jnp.float32(samples['log_r_c']),
                       'log_r_s': jnp.float32(samples['log_r_s']),
                       'beta': jnp.float32(samples['beta']),
                       'epsilon': jnp.float32(samples['epsilon'])}}

        sb_sample = jax.vmap(lambda p: self.sb.apply(p, self.bin_center, self.nh))(params)

        median_params = {'xray_surface_brightness':
                             {'log_bkg': jnp.float32(self.mean_model.posterior_median['log_bkg'])},
                         'xray_surface_brightness/~/xray_emissivity/~/clean_vikhlinin_model':
                             {'log_ne2': jnp.float32(self.mean_model.posterior_median['log_ne2']),
                              'log_r_c': jnp.float32(self.mean_model.posterior_median['log_r_c']),
                              'log_r_s': jnp.float32(self.mean_model.posterior_median['log_r_s']),
                              'beta': jnp.float32(self.mean_model.posterior_median['beta']),
                              'epsilon': jnp.float32(self.mean_model.posterior_median['epsilon'])}}

        plt.figure(figsize=(9, 6))

        # Plot the envelope of the surface brightness
        plt.fill_between(self.bin_center,
                         jnp.percentile(sb_sample, 14, axis=0),
                         jnp.percentile(sb_sample, 86, axis=0),
                         color='orangered',
                         alpha=0.25)

        # Plot the measured surface brightness
        plt.errorbar(self.bin_center,
                     self.surface_brightness.pdf_median(),
                     xerr=self.bin_width,
                     yerr=self.e_surface_brightness,
                     fmt='None',
                     color='black',
                     label='Measured')

        # Plot the median surface brightness
        plt.plot(self.bin_center,
                 self.sb.apply(median_params, self.bin_center, self.nh),
                 color='red',
                 label='Model')

        # Plot the measured background
        plt.plot(self.bin_center, self.bkg / self.exp,
                 color='green',
                 label='Background')

        # Format the plot
        plt.xlim(left=min(self.bin_center), right=(max(self.bin_center)))
        plt.ylim(top=max(self.surface_brightness.pdf_median()) * 2, bottom=min(self.bkg.pdf_median() / self.exp) / 2)
        plt.xlabel('Radius [$R_{500}$]')
        plt.ylabel('Surface Brightness [Counts kpc$^{-2}$ s$^{-1}$]')

        plt.loglog()
        plt.legend()
