from __future__ import annotations

import numpy as np
import astropy.units as u
import copy
import os 
from regions import Regions
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from regions import SkyRegion
from astropy.nddata import block_reduce
from astropy.cosmology import FlatLambdaCDM, Cosmology
from ..utils.config import config


class Cluster:
    """
    Data container for a cluster observation/mosaic as defined by pyproffit.

    Attributes:
        img (np.ndarray): Image data
        exp (np.ndarray): Exposure map
        bkg (np.ndarray): Background map
        nh (np.ndarray): Hydrogen column density map
        wcs (astropy.wcs.WCS): WCS object
        header (astropy.io.fits.Header): Header object
        degree_per_pixel (astropy.units.Quantity): Angular size of a pixel in degrees
        kpc_per_pixel (astropy.units.Quantity): Angular size of a pixel in kpc
        shape (tuple): Shape of the image
        x_c (float): X coordinate of the cluster center
        y_c (float): Y coordinate of the cluster center
        y_ref (np.ndarray): Y coordinate of the image
        x_ref (np.ndarray): X coordinate of the image
        coords (astropy.coordinates.SkyCoord): Coordinates of the image
        z (float): Redshift of the cluster
        r_500 (astropy.units.Quantity): Radius of the cluster at 500 times the critical density.
        t_500 (astropy.units.Quantity): Angular radius of the cluster at 500 times the critical density.
        center (astropy.coordinates.SkyCoord): Coordinates of the cluster center
        name (str): Name of the cluster
        imglink (str): Path to the image file
        explink (str): Path to the exposure map file
        bkglink (str): Path to the background map file
        cosmo (astropy.cosmology.FlatLambdaCDM): Cosmology used to compute the angular and physical sizes
        regions (regions.Regions): Regions to exclude from the analysis
        unmasked_img (np.ndarray): Unmasked image data
        unmasked_exposure (np.ndarray): Unmasked exposure map
    """

    cosmo: Cosmology

    def __init__(self, imglink: str,
                 explink: str=None,
                 bkglink: str=None,
                 reglink: str=None,
                 nhlink: str=None,
                 redshift: float=0.,
                 r_500: u.Quantity|None=None,
                 t_500: u.Quantity|None=None,
                 ra: float|u.Quantity=None,
                 dec: float|u.Quantity=None,
                 name: str=None):
        r"""
        Constructor for the Cluster class.

        Parameters:
            imglink (str): Path to the image file
            explink (str): Path to the exposure map file
            bkglink (str): Path to the background map file
            redshift (float): Redshift of the cluster
            r_500 (astropy.units.Quantity): Radius of the cluster at 500 times the critical density.  Either r_500 or t_500 must be provided.
            t_500 (astropy.units.Quantity): Angular radius of the cluster at 500 times the critical density.  Either r_500 or t_500 must be provided.
            ra (float|astropy.units.Quantity): Right ascension of the cluster center
            dec (float|astropy.units.Quantity): Declination of the cluster center
            name (str): Name of the cluster
        """

        self.img = fits.getdata(imglink)
        self.imglink = imglink
        self.explink = explink
        self.bkglink = bkglink
        self.cosmo = FlatLambdaCDM(70, 0.3)
        self.z = redshift
        self.name = name

        if r_500 is not None:

            self.r_500 = r_500
            self.t_500 = (r_500/self.cosmo.kpc_proper_per_arcmin(self.z)).to(u.arcmin)

        elif t_500 is not None:

            self.t_500 = t_500
            self.r_500 = (t_500*self.cosmo.kpc_proper_per_arcmin(self.z)).to(u.kpc)

        self.center = SkyCoord(ra=ra, dec=dec, unit='degree')

        head = fits.getheader(imglink)
        self.header = head
        self.wcs = WCS(head, relax=False)

        if 'CDELT2' in head:
            self.degree_per_pixel = head['CDELT2'] * u.deg / u.pix
        elif 'CD2_2' in head:
            self.degree_per_pixel = head['CD2_2'] * u.deg / u.pix

        self.kpc_per_pixel = self.cosmo.kpc_proper_per_arcmin(self.z) * self.degree_per_pixel
        self.kpc_per_pixel = self.kpc_per_pixel.to(u.kpc / u.pixel)
        self.shape = self.img.shape
        self.exp = fits.getdata(explink, memmap=False) if explink is not None else np.ones(self.shape)
        self.bkg = fits.getdata(bkglink, memmap=False) if bkglink is not None else np.zeros(self.shape)
        
        self.exp *= self.kpc_per_pixel.value ** 2
        self.unmasked_img = copy.deepcopy(self.img)
        self.img *= (self.exp>0)
        
        self.x_c, self.y_c = self.center.to_pixel(self.wcs)
        self.y_ref, self.x_ref = np.indices(self.shape)
        self.coords = SkyCoord.from_pixel(self.x_ref, self.y_ref, self.wcs)

        self.load_nh(nhlink)
        self.region(reglink)

    @classmethod
    def from_catalog_row(cls, row):
        """
        DEPRECATED: Build a cluster object from a row in the X-COP catalog
        """

        name = row['NAME']
        
        imglink = os.path.join(config.DATA_PATH, f'XCOP/{name}/mosaic_{name.lower()}.fits.gz')
        explink = os.path.join(config.DATA_PATH, f'XCOP/{name}/mosaic_{name.lower()}_expo.fits.gz')
        bkglink = os.path.join(config.DATA_PATH, f'XCOP/{name}/mosaic_{name.lower()}_bkg.fits.gz')
        
        instance = cls(imglink,
                       explink=explink,
                       bkglink=bkglink,
                       redshift=row['REDSHIFT'],
                       r_500=row['R500_HSE'] * u.kpc,
                       ra=row['RA'],
                       dec=row['DEC'],
                       name=name)

        instance.region(os.path.join(config.DATA_PATH, f'XCOP/{name}/src_ps.reg'))
        instance.load_nh(os.path.join(config.DATA_PATH, f'XCOP/{name}/{name}_nh.fits'))

        return instance

    def region(self, region_file):
        """
        Filter out regions provided in an input DS9 region file

        Parameters:
            region_file (str): Path to region file. Accepted region file formats are fk5 and image.
        """

        regions = Regions.read(region_file)
        mask = np.ones(self.shape)*(self.exp>0.)

        for region in regions:

            if isinstance(region, SkyRegion):
                # Turn a sky region into a pixel region
                region = region.to_pixel(self.wcs)

            mask[region.to_mask().to_image(self.shape) > 0] = 0

        #print('Excluded %d sources' % (len(regions)))
        self.unmasked_exposure = copy.deepcopy(self.exp)
        self.img = self.img*mask
        self.exp = self.exp*mask
        self.regions = regions

    def load_nh(self, nh_file):
        """
        Load the hydrogen column density map

        Parameters:
            nh_file (str): Path to the hydrogen column density map file
        """

        nh = fits.getdata(nh_file)
        assert nh.shape == self.shape
        self.nh = nh

    def voronoi(self, voronoi_file, rebin_factor=5, exclusion=1, t_500_percent=1.):
        """
        Load the voronoi binning map. It must be produced using the vorbin package.
        See https://pypi.org/project/vorbin/ for more information and especially
        the scripts/voronoi.ipynb notebook for an example of how to generate the map.

        Rebin the data using the previously loaded voronoi map.
        It assumes that the Voronoi binning algorithm has been run on the data with a
        first rough 4x4 rebinning, which I used to accelerate the computation of maps.

        !!! danger
            This function contains a lot of hard-coded values that should be changed.
            It requires the user to precisely remember how the data was processed before
            using the `vorbin`package and try to applies the same to the untouched cluster
            data.

        Parameters:
            voronoi_file (str): Path to the voronoi map file
            rebin_factor (int): Rebinning factor
            exclusion (float): factor used in `reduce_to_r500`
            t_500_percent (float): Radius of the cluster in units of t_500 when selecting pixels

        Returns:
            cluster: A new Cluster object with the rebinned data.

        !!! warning
            This function does not act in place, and instead return a new object.
        """

        new_data = self.reduce_to_r500(exclusion).rebin(rebin_factor)

        new_data.voronoi = np.loadtxt(voronoi_file)

        indexes = (new_data.exp > 0) & (new_data.coords.separation(new_data.center) < new_data.t_500*t_500_percent)
        y_ref, x_ref = new_data.y_ref[indexes], new_data.x_ref[indexes]
        img = np.array(new_data.img[indexes].astype(int))
        exp = np.array(new_data.exp[indexes].astype(np.float32))
        
        img *= (exp > 0.)
        bkg = np.array(new_data.bkg[indexes].astype(np.float32))
        nH = np.array(new_data.nh[indexes].astype(np.float32))
        bin_number = np.array(new_data.voronoi.astype(int))

        unique_bin = np.unique(bin_number)
        x_ref_reduced = np.empty_like(unique_bin, dtype=np.float32)
        y_ref_reduced = np.empty_like(unique_bin, dtype=np.float32)
        img_reduced = np.empty_like(unique_bin)
        exp_reduced = np.empty_like(unique_bin, dtype=np.float32)
        bkg_reduced = np.empty_like(unique_bin)
        nH_reduced = np.empty_like(unique_bin, dtype=np.float32)

        for i, number in enumerate(unique_bin):
            bin_index = (bin_number == number)

            x_ref_reduced[i] = np.mean(x_ref[bin_index])
            y_ref_reduced[i] = np.mean(y_ref[bin_index])
            nH_reduced[i] = np.mean(nH[bin_index])
            img_reduced[i] = np.sum(img[bin_index])
            exp_reduced[i] = np.sum(exp[bin_index])
            bkg_reduced[i] = np.sum(bkg[bin_index])

        new_data.img = img_reduced
        new_data.exp = exp_reduced
        new_data.bkg = bkg_reduced
        new_data.nh = nH_reduced
        new_data.x_ref = x_ref_reduced
        new_data.y_ref = y_ref_reduced
        new_data.shape = unique_bin.shape

        return new_data

    def flatten(self, r_500_percent: float=1.):
        """
        Flatten the data in a 1D array and remove pixels outside the specified radius.
        It also removes pixels with no exposure.

        Parameters:
            r_500_percent (float): Radius of the cluster in units of r_500

        Returns:
            cluster: A new Cluster object with the flattened data.

        !!! warning
            This function does not act in place, and instead return a new object.
        """
        

        new_data = copy.deepcopy(self)
        index = (new_data.exp > 0) & (new_data.center.separation(new_data.coords) < new_data.t_500 * r_500_percent)
        new_data.img = new_data.img[index]
        new_data.exp = new_data.exp[index]
        new_data.bkg = new_data.bkg[index]
        new_data.nh = new_data.nh[index]
        new_data.shape = new_data.img.shape
        
        new_data.y_ref, new_data.x_ref = new_data.y_ref[index], new_data.x_ref[index]

        new_data.index = index 
        
        return new_data

    def rebin(self, factor: int):
        """
        Rebin the data by a factor of `factor`. It uses the astropy block_reduce function.

        Parameters:
            factor (int): Rebinning factor

        Returns:
            cluster: A new Cluster object with the rebinned data.

        !!! warning
            This function does not act in place, and instead return a new object.
        """

        def sum_reduce(vec, factor):
            
            return np.nan_to_num(block_reduce(np.where(self.exp > 0, vec, np.nan), factor))
        
        def sum_reduce_unmasked(vec, factor):
            
            return np.nan_to_num(block_reduce(np.where(self.unmasked_exposure > 0, vec, np.nan), factor))
        
        def mean_reduce(vec, factor):
            
            return np.nan_to_num(block_reduce(np.where(self.exp > 0, vec, np.nan), factor, np.mean))
        
        new_data = copy.deepcopy(self)
        new_data.wcs = new_data.wcs[::factor, ::factor]
        new_data.img = sum_reduce(self.img, factor)
        new_data.exp = sum_reduce(self.exp, factor)
        new_data.bkg = sum_reduce(self.bkg, factor)
        new_data.nh = mean_reduce(self.nh, factor)
        new_data.unmasked_img = sum_reduce_unmasked(self.unmasked_img, factor)
        new_data.unmasked_exposure = sum_reduce_unmasked(self.unmasked_exposure, factor)
        new_data.shape = new_data.img.shape
        new_data.kpc_per_pixel *= factor
        new_data.degree_per_pixel *= factor

        new_data.x_c, new_data.y_c = new_data.center.to_pixel(new_data.wcs)
        new_data.y_ref, new_data.x_ref = np.indices(new_data.shape)
        new_data.coords = SkyCoord.from_pixel(new_data.x_ref, new_data.y_ref, new_data.wcs)

        return new_data

    def reduce_fov(self, mean_model, r500_cut):
        """
        Reduce the field of view to a given radius in units of r_500, using the best-fit
        mean model, which includes ellipticity for the cluster shape.

        Parameters:
            mean_model (MeanModel): Best-fit mean model
            r500_cut (float): Radius in units of r_500

        Returns:
            cluster: A new Cluster object with the reduced field of view.

        !!! warning
            This function does not act in place, and instead return a new object.
        """

        
        rows = np.any(mean_model.rad<r500_cut, axis=1)
        cols = np.any(mean_model.rad<r500_cut, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        
        new_data = copy.deepcopy(self)
        new_data.wcs = new_data.wcs[rmin:rmax, cmin:cmax]
        new_data.img = self.img[rmin:rmax, cmin:cmax]
        new_data.exp = self.exp[rmin:rmax, cmin:cmax]
        new_data.bkg = self.bkg[rmin:rmax, cmin:cmax]
        new_data.nh = self.nh[rmin:rmax, cmin:cmax]
        new_data.unmasked_exposure = self.unmasked_exposure[rmin:rmax, cmin:cmax]
        new_data.unmasked_img = self.unmasked_img[rmin:rmax, cmin:cmax]
        new_data.shape = new_data.img.shape

        new_data.x_c, new_data.y_c = new_data.center.to_pixel(new_data.wcs)
        new_data.y_ref, new_data.x_ref = np.indices(new_data.shape)
        new_data.coords = SkyCoord.from_pixel(new_data.x_ref, new_data.y_ref, new_data.wcs)

        return new_data

    def reduce_to_r500(self, r500_cut=1.):
        """
        Reduce the field of view to a given radius in units of r_500. This does not
        take into account the ellipticity of the cluster, and simply assumes a circular
        symetry and uses the first proposed center to compute the radius.

        Parameters:
            r500_cut (float): Radius in units of r_500

        Returns:
            cluster: A new Cluster object with the reduced field of view.

        !!! warning
            This function does not act in place, and instead return a new object.
        """

        valid = (self.exp>0)&(self.coords.separation(self.center) < self.t_500*r500_cut)
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]


        new_data = copy.deepcopy(self)
        new_data.wcs = new_data.wcs[rmin:rmax, cmin:cmax]
        new_data.img = self.img[rmin:rmax, cmin:cmax]
        new_data.exp = self.exp[rmin:rmax, cmin:cmax]
        new_data.bkg = self.bkg[rmin:rmax, cmin:cmax]
        new_data.nh = self.nh[rmin:rmax, cmin:cmax]
        new_data.unmasked_exposure = self.unmasked_exposure[rmin:rmax, cmin:cmax]
        new_data.unmasked_img = self.unmasked_img[rmin:rmax, cmin:cmax]
        new_data.shape = new_data.img.shape

        new_data.x_c, new_data.y_c = new_data.center.to_pixel(new_data.wcs)
        new_data.y_ref, new_data.x_ref = np.indices(new_data.shape)
        new_data.coords = SkyCoord.from_pixel(new_data.x_ref, new_data.y_ref, new_data.wcs)

        return new_data

    def plot_cluster(self):
        """
        Helper function to plot the observation components
        """

        import matplotlib.pyplot as plt
        import cmasher as cmr
        from matplotlib.colors import LogNorm

        fig, axs = plt.subplots(
            figsize=(12, 5),
            nrows=1,
            ncols=3,
            subplot_kw={'projection': self.wcs}
        )

        img_plot = axs[0].imshow(
            np.where(self.exp > 0, self.img, np.nan),
            cmap=cmr.cosmic,
            norm=LogNorm(vmin=0.1)
        )

        exp_plot = axs[1].imshow(
            np.where(self.exp > 0, self.exp, np.nan),
            cmap=cmr.ember
        )

        bkg_plot = axs[2].imshow(
            np.where(self.exp > 0, self.bkg, np.nan),
            cmap=cmr.cosmic
        )

        plt.colorbar(
            mappable=img_plot,
            ax=axs[0],
            location='bottom',
            label="Image (Photons)"
        )

        plt.colorbar(
            mappable=exp_plot,
            ax=axs[1],
            location='bottom',
            label=r"Effective exposure (seconds $\times$ kpc$^2$)"
        )

        plt.colorbar(
            mappable=bkg_plot,
            ax=axs[2],
            location='bottom',
            label="Background (Photons)"
        )

        plt.tight_layout()
