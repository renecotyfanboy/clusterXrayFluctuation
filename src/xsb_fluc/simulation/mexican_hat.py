import numpy as np
import jax.numpy as jnp
import jax.numpy.fft as fft
import haiku as hk
import astropy.units as u
from xsb_fluc.data.cluster import Cluster


class MexicanHatMask(hk.Module):
    """
    Auxiliary class for the MexicanHat class. Handle the convolution in Fourier space,
    the padding, etc.
    """
    
    def __init__(self, shape, mask=None, pixsize=0.005):
        super(MexicanHatMask, self).__init__()
        
        self.pixsize = pixsize 
        self.epsilon = 1e-3 
        self.pad_size = 50
        self.mask = jnp.pad(mask, self.pad_size)
        
        kx = fft.fftfreq(shape[0] + 2*self.pad_size, d=self.pixsize)
        ky = fft.rfftfreq(shape[1]+ 2*self.pad_size, d=self.pixsize)
        KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
        self.K = (KX**2 + KY**2)**(1/2)
        
    @classmethod
    def from_data(cls, data: Cluster, mask=None):

        if mask is None:
            mask = data.exp > 0.

        return cls(mask.shape, mask=mask, pixsize=(data.kpc_per_pixel/data.r_500).to(1/u.pixel).value)
    
    def fourier_kernel(self, sigma):
    
        return jnp.exp(-2*(jnp.pi*self.K*sigma)**2)

    def gaussian_blur(self, to_blur, sigma):

        blurred_img = fft.irfft2(fft.rfft2(to_blur)*self.fourier_kernel(sigma), s=to_blur.shape)
        return blurred_img#[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

    def filter(self, img, sigma):
        
        img_masked = jnp.pad(img, self.pad_size) * self.mask
        gsigma1 = self.gaussian_blur(img_masked, sigma*(1+self.epsilon)**(-1/2))
        gsigma2 = self.gaussian_blur(img_masked, sigma*(1+self.epsilon)**(+1/2))
        # FFT-convolve mask with the two scales
        gmask1 = self.gaussian_blur(self.mask, sigma*(1+self.epsilon)**(-1/2))
        gmask2 = self.gaussian_blur(self.mask, sigma*(1+self.epsilon)**(+1/2))
        # Eq. 6 of Arevalo et al. 2012
        
        r1 = jnp.where(gmask1 != 0., gsigma1/gmask1, 0.)
        r2 = jnp.where(gmask2 != 0., gsigma2/gmask2, 0.)
        
        return (r1 - r2)*self.mask

    def __call__(self, img, scale):

        k = 1/scale
        
        #Arévalo & al A5
        sigma = 1 / jnp.sqrt(2 * jnp.pi ** 2) / k
        convolved_img = self.filter(img, sigma)
        
        return convolved_img


class MexicanHat(hk.Module):
    """
    Mexican Hat filter for the power spectrum, using the implementation of Arévalo et al. 2012.
    """
    
    def __init__(self, data, mask=None):
        """
        Constructor for the MexicanHat class.

        Parameters:
            data (Cluster): Cluster object containing the data.
            mask (np.ndarray): Mask to apply to the data.

        !!! note
            The `PowerSpectrum`class is a `Cluster` oriented wrapper around this class.
        """
        super(MexicanHat, self).__init__()
        self.mexican_hat = MexicanHatMask.from_data(data, mask=mask)
        self.pixsize = self.mexican_hat.pixsize
        self.epsilon = self.mexican_hat.epsilon
        self.ratio = 1./np.sum(mask)
        self.Y = jnp.pi 

    def __call__(self, img, scale):
        """
        Computes the Mexican Hat filter for a given scale.

        Parameters:
            img (np.ndarray): Image to filter.
            scale (float): Scale of the filter.
        """

        img_convolved = self.mexican_hat(img, scale)
        k = 1/scale

        return jnp.sum(img_convolved**2)/(self.epsilon**2*self.Y*k**2)*self.ratio#/self.pixsize**2
