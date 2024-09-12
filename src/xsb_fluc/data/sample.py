import os
import warnings
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join
from src.xsb_fluc.utils.config import config
from src.xsb_fluc.data.cluster import Cluster
from abc import ABC, abstractmethod


class ClusterSample(ABC):

    def __iter__(self):

        for name in self.names:

            yield self.cluster_from_name(name)

    def __getitem__(self, name):

        return self.cluster_from_name(name)

    def inspect(self):

        for cluster in self:

            idx = cluster.exp>0

            plt.figure(figsize=(10,10))
            plt.title(cluster.name)
            plt.subplot(131)
            plt.imshow(np.where(idx, cluster.img, np.nan))
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(np.where(idx, cluster.exp, np.nan))
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(np.where(idx, cluster.bkg, np.nan))
            plt.axis('off')
            plt.show()

    @property
    @abstractmethod
    def master_table(self):
        pass

    @property
    @abstractmethod
    def names(self):
        pass

    @abstractmethod
    def cluster_from_name(self, name):
        pass


class XCOP(ClusterSample):

    with warnings.catch_warnings():
        # Supposed to hide FITS deprecated keyword warnings for the X-COP clusters
        warnings.simplefilter("ignore")
        master_table = Table.read(os.path.join(config.DATA_PATH, 'XCOP/XCOP_master_table.fits'))

    names = list(master_table['NAME'])
    names.remove('HydraA')

    def cluster_from_name(self, name):

        row = self.master_table[self.master_table['NAME'] == name][0]
        imglink = os.path.join(config.DATA_PATH, f'XCOP/{name}/mosaic_{name.lower()}.fits.gz')
        explink = os.path.join(config.DATA_PATH, f'XCOP/{name}/mosaic_{name.lower()}_expo.fits.gz')
        bkglink = os.path.join(config.DATA_PATH, f'XCOP/{name}/mosaic_{name.lower()}_bkg.fits.gz')

        with warnings.catch_warnings():
            # Supposed to hide FITS deprecated keyword warnings for the X-COP clusters
            warnings.simplefilter("ignore")

            cluster = Cluster(imglink,
                           explink=explink,
                           bkglink=bkglink,
                           redshift=row['REDSHIFT'],
                           r_500=row['R500_HSE'] * u.kpc,
                           ra=row['RA'],
                           dec=row['DEC'],
                           name=name)

        cluster.region(os.path.join(config.DATA_PATH, f'XCOP/{name}/src_ps.reg'))
        cluster.load_nh(os.path.join(config.DATA_PATH, f'XCOP/{name}/{name}_nh.fits'))

        return cluster

class CHEXMATE(ClusterSample):

    master_table = Table.read(os.path.join(config.DATA_PATH, 'master_chexmate.fits'), 1)
    _master_table_nh = Table.read(os.path.join(config.DATA_PATH, 'master_chexmate.fits'), 2)
    _master_table_nh.keep_columns(['NHTOT', 'HER_INDEX'])



    master_table['NAME'] = [name.strip().replace(" ", "") for name in master_table['NAME']]
    names = [x for x in os.listdir(os.path.join(config.DATA_PATH, 'CHEXMATE')) if x[0:5]=='PSZ2G']

    with warnings.catch_warnings():
        # Supposed to hide FITS deprecated keyword warnings for the X-COP clusters
        warnings.simplefilter("ignore")
        master_table = join(master_table, _master_table_nh, keys='HER_INDEX')

    def cluster_from_name(self, name):

        if name not in self.names :
            raise NameError(f'{name} not in this sample')

        inter = os.listdir(os.path.join(config.DATA_PATH, f'CHEXMATE/{name}'))

        if len(inter)>1:

            inter = 'mosaic'

        else:

            inter=inter[0]

        path_to_img = os.path.join(config.DATA_PATH, f'CHEXMATE/{name}/{inter}/images_700_1200/')
        path_to_ewav = os.path.join(config.DATA_PATH, f'CHEXMATE/chexmate_ewav_regions/{name}/{inter}/ewav_regions/')

        row = self.master_table[self.master_table['NAME'] == name][0]

        imglink = os.path.join(path_to_img, 'epic-obj-im-700-1200.fits.gz')
        explink = os.path.join(path_to_img, 'epic-exp-im-700-1200.fits.gz')
        bkglink = os.path.join(path_to_img, 'epic-back-tot-sp-700-1200.fits.gz')

        with warnings.catch_warnings():
            # Supposed to hide FITS deprecated keyword warnings for the X-COP clusters
            warnings.simplefilter("ignore")

            cluster = Cluster(imglink,
                           explink=explink,
                           bkglink=bkglink,
                           redshift=row['REDSHIFT'],
                           t_500=row['R500'] * u.arcmin,
                           ra=row['RA'],
                           dec=row['DEC'],
                           name=name)

        cluster.region(os.path.join(config.DATA_PATH, f'CHEXMATE/{name}/{inter}', 'regions', f'srclist_{name}.reg'))
        cluster.nh = fits.getdata(os.path.join(path_to_img, f'{name}_nh.fits'))
        cluster.thresh_soft = Table.read(os.path.join(path_to_ewav, 'srclist_300_2000_filtered.fits'))['RATE'].min()
        cluster.thresh_hard = Table.read(os.path.join(path_to_ewav, 'srclist_2000_7000_filtered.fits'))['RATE'].min()

        return cluster
    
    def interactive_m_z(self):
        
        import plotly.express as px
        fig = px.scatter(x=self.master_table['M500'], y=self.master_table['REDSHIFT'], hover_name=self.master_table['NAME'])
        fig.show()


xcop = XCOP()
chexmate = CHEXMATE()