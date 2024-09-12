import numpy as np
import warnings
from jax.random import PRNGKey
from copy import deepcopy

def set_params(ref, params):

    new_params = deepcopy(ref)#copy()

    for sub_dict in new_params.values():

        for key, values in sub_dict.items():
            sub_dict[key] = params[key]

    return new_params


def rng_key():

    return PRNGKey(np.random.randint(0, int(1e6)))

def hide_warnings(f):
    
    def new_f(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = f(*args, **kwargs)
        return res 

    return new_f

@hide_warnings
def inspect_cluster(name):
    
    from jax.config import config
    config.update("jax_enable_x64", True)
    from src.xsb_fluc.data.sample import chexmate
    import numpy as np
    from astropy.table import Table
    from src.xsb_fluc.utils.subsamples import list_of_cluster
    import arviz as az
    from src.xsb_fluc.data.mean_model import MeanModel
    import cmasher as cmr
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from matplotlib.colors import SymLogNorm, LogNorm

    morpho_table = Table.read('data/morpho_chexmate.fits')
    list_of_cluster['all']
    levels = np.array([1.])
    
            
    line = morpho_table[morpho_table['Name'] == name[4:]][0]
    inference_data = az.from_netcdf(f'results/power_spectrum/{name}/abs_0_1.posterior')
    power_spectrum = az.extract(inference_data)
    np.array(power_spectrum.theta).T
    

    cluster = chexmate[name]
    data = cluster.rebin(3)
    mean_model = MeanModel.from_data(data)
    reduced_data = data.reduce_fov(mean_model, 1.)
    MeanModel.from_data(reduced_data)

    img = np.where(mean_model.data.exp>0, mean_model.fluctuation_absolute, np.nan)
    rad = mean_model.rad

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    ax = plt.gca()
    im = ax.imshow(np.where(data.exp>0, gaussian_filter(data.img, 1.5), np.nan),
                    cmap=cmr.cosmic, #cmr.fusion
                    norm = LogNorm(vmin=0.1, vmax=100))
    ax.contour(rad, levels, colors='white', alpha=1.)
    plt.colorbar(mappable=im)
    plt.title('IMG')
    ax.axis('off')

    plt.subplot(122)
    ax = plt.gca()
    im = ax.imshow(np.where(data.unmasked_exposure>0, gaussian_filter(data.unmasked_img, 1.5), np.nan),
                    cmap=cmr.cosmic, #cmr.fusion
                    norm = LogNorm(vmin=0.1, vmax=100))
    
    ax.contour(rad, levels, colors='white', alpha=1.)
    plt.colorbar(mappable=im)
    plt.title('IMG UNMASKED')
    ax.axis('off')

    plt.show();

    plt.figure(figsize=(15,15))
    ax = plt.gca()
    im = ax.imshow(np.where(np.isnan(img),
                            np.nan,
                            gaussian_filter(np.nan_to_num(img),1.5)),
                    cmap=cmr.guppy, #cmr.fusion
                    norm = SymLogNorm(linthresh=1e-8, vmin=-5e-6, vmax=5e-6),
                    alpha=np.where((rad>1)|(rad<0.15), 0.2, 1))
    ax.contour(rad, levels, colors='white', alpha=1.)
    plt.colorbar(mappable=im)
    ax.axis('off')
    plt.title(f'Fluctuations \n {name}({line["state"]}, M={line["M"]})\n R = {cluster.r_500.value:.0f}, z = {cluster.z:.2f}')
    plt.show();

    #az.plot_trace(mean_model.inference_data)
    #plt.suptitle('Mean model parameters');
    #plt.show();
    samples = np.loadtxt(f'results/power_spectrum/{name}/samples.txt')
    az.plot_pair({'log_sigma': samples[:, 0], 'log_inj': samples[:, 1], 'alpha': samples[:, 2]}, kind='hexbin');
    plt.suptitle('Turbulent parameters');
    plt.show();