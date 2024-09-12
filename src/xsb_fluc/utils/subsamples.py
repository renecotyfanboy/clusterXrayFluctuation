import os
import numpy as np
from astropy.table import Table
import pandas as pd
from ..utils.config import config
from ..data.sample import chexmate


def equalObs(x, nbin):
    return np.interp(np.linspace(0, len(x), nbin + 1), np.arange(len(x)), np.sort(x))


list_of_cluster = {}
morpho_table = Table.read(os.path.join(config.DATA_PATH, 'morpho_chexmate.fits'))

################################## Filter all problematic clusters

excluded = ['PSZ2G028.63+50.15',  # Clusters from "problematic" list
            'PSZ2G283.91+73.87',  # this one is in problematic list and has low counts
            'PSZ2G285.63+72.75',
            'PSZ2G325.70+17.34',
            'PSZ2G067.17+67.46',
            'PSZ2G021.10+33.24',
            'PSZ2G028.89+33.24',
            'PSZ2G107.10+65.32',
            'PSZ2G172.98-53.55',
            'PSZ2G040.03+74.95N',  # Also remove double
            'PSZ2G057.78+52.32N',
            'PSZ2G068.22+15.18E',
            'PSZ2G042.81+56.61NW']

excluded += ['PSZ2G040.03+74.95',  # Amas double sur la ligne de visée
             # https://ui.adsabs.harvard.edu/abs/2010AstBu..65..205K/abstract
             # https://ui.adsabs.harvard.edu/abs/2006AstL...32...84K/abstract
             # https://iopscience.iop.org/article/10.1088/0004-6256/147/6/156/pdf
             'PSZ2G068.22+15.18',  # Cluster CIZA
             'PSZ2G080.37+14.64',
             'PSZ2G313.88-17.11',
             'PSZ2G085.98+26.69',   # I don't like it and he is at M=0.5
             'PSZ2G042.81+56.61',    # I don't like it either
             'PSZ2G238.69+63.26',
             'PSZ2G008.31-64.74', # Unadapted mean model with huge positive residual
]

XCOP_in_PSZ2 = ['PSZ2G075.71+13.51', #A2319
                'PSZ2G304.91+45.43', #A1644
                'PSZ2G033.81+77.18', #A1795
                'PSZ2G115.25-72.07', #A85
                'PSZ2G229.93+15.30', #A644
                'PSZ2G006.49+50.56', #A2029
                'PSZ2G044.20+48.66', #A2142
                'PSZ2G093.92+34.92', #A2255
                'PSZ2G265.02-48.96', #A3158
                'PSZ2G272.08-40.16', #A3266
                'PSZ2G058.29+18.55', #RXC
                #ZW ??
                ]

# Clusters with high dynamical state
M_excluded = ['PSZ2' + name for name, w in zip(morpho_table['Name'], morpho_table['w']) if w > 0.2]

# Cluster with successful analysis and not excluded
all_names = [name for name in chexmate.master_table['NAME'] if
             (os.path.exists(os.path.join(config.RESULTS_PATH, f'power_spectrum/{name}/abs_0_1.posterior'))) if
             (name not in excluded)]
reduced_names = [name for name in all_names if (name not in M_excluded)]
reduced_master_table = chexmate.master_table[[x['NAME'] in reduced_names for x in chexmate.master_table]]

################################## Dynamical state
list_of_cluster['relaxed'] = ['PSZ2' + name for name in morpho_table[morpho_table['state'] == 'R']['Name'] if
                              'PSZ2' + name in all_names]
list_of_cluster['mixed'] = ['PSZ2' + name for name in morpho_table[morpho_table['state'] == 'M']['Name'] if
                            'PSZ2' + name in all_names]
list_of_cluster['disturbed'] = ['PSZ2' + name for name in morpho_table[morpho_table['state'] == 'D']['Name'] if
                                'PSZ2' + name in all_names]
list_of_cluster['all'] = all_names
list_of_cluster['reduced'] = reduced_names
list_of_cluster['excluded'] = excluded

################################## Tier 1 and 2
tier_1 = reduced_master_table[[(x['TIER'] == 1) or (x['TIER'] == 12) for x in reduced_master_table]]
tier_2 = reduced_master_table[[(x['TIER'] == 2) or (x['TIER'] == 12) for x in reduced_master_table]]
list_of_cluster['tier_1'] = list(tier_1['NAME'])
list_of_cluster['tier_2'] = list(tier_2['NAME'])

################################## Define equal frequency xsb_fluc bins

mass = tier_1['M500']
redshift = tier_2['REDSHIFT']
size = reduced_master_table['R500']
state = np.asarray([morpho_table[morpho_table['Name'] == name[4:]][0]['w'] for name in reduced_names])

_, bins_mass = np.histogram(mass, equalObs(mass, 3))
_, bins_redshift = np.histogram(redshift, equalObs(redshift, 3))
_, bins_size = np.histogram(size, equalObs(size, 3))
_, bins_state = np.histogram(state, equalObs(state, 3))

for i in range(len(bins_mass) - 1):
    list_of_cluster[f'mass_{i}'] = list(tier_1[(bins_mass[i] <= mass) & (mass <= bins_mass[i + 1])]['NAME'])

for i in range(len(bins_redshift) - 1):
    list_of_cluster[f'redshift_{i}'] = list(
        tier_2[(bins_redshift[i] <= redshift) & (redshift <= bins_redshift[i + 1])]['NAME'])

for i in range(len(bins_size) - 1):
    list_of_cluster[f'size_{i}'] = list(
        reduced_master_table[(bins_size[i] <= size) & (size <= bins_size[i + 1])]['NAME'])

for i in range(len(bins_state) - 1):
    list_of_cluster[f'state_{i}'] = list(
        reduced_master_table[(bins_state[i] <= state) & (state <= bins_state[i + 1])]['NAME'])

# Build the mass & redshift without disturbed

for i in range(len(bins_mass) - 1):
    list_of_cluster[f'mass_{i}_without_D'] = [x for x in list_of_cluster[f'mass_{i}'] if
                                              x not in list_of_cluster['disturbed']]

for i in range(len(bins_mass) - 1):
    list_of_cluster[f'redshift_{i}_without_D'] = [x for x in list_of_cluster[f'redshift_{i}'] if
                                                  x not in list_of_cluster['disturbed']]

for i in range(len(bins_mass) - 1):
    list_of_cluster[f'size_{i}_without_D'] = [x for x in list_of_cluster[f'size_{i}'] if
                                              x not in list_of_cluster['disturbed']]

list_of_cluster['temp_fluc'] = ['PSZ2' + x for x in
                                Table.read(os.path.join(config.DATA_PATH, 'CHEXMATE/lorenzo_list.csv'))['Name'] if
                                'PSZ2' + x in list_of_cluster['reduced']]
list_of_cluster['with_radio_halo'] = [x for x in
                                      Table.read(os.path.join(config.DATA_PATH, 'CHEXMATE/with_radio_halo.csv'))['Name']
                                      if x in list_of_cluster['reduced']]

list_of_cluster['no_radio_halo'] = [x for x in Table.read(
    os.path.join(config.DATA_PATH, 'CHEXMATE/no_radio_halo.csv'))['Name'] if x in list_of_cluster['reduced']]


df = pd.read_csv(os.path.join(config.DATA_PATH, 'CHEXMATE/gradient.csv'))

list_of_cluster['gradient_0'] = [x for x in list(df[df['mean_scale_height']<-0.8]['name']) if x in list_of_cluster['reduced']]
list_of_cluster['gradient_1'] = [x for x in list(df[df['mean_scale_height']>-0.8]['name']) if x in list_of_cluster['reduced']]

df = pd.read_csv(os.path.join(config.DATA_PATH, 'CHEXMATE/spatial_resolution.csv'))

_, bins_resolution = np.histogram(df['res'], equalObs(df['res'], 3))

for i in range(len(bins_resolution) - 1):
    list_of_cluster[f'res_{i}'] = list(df[(bins_resolution[i] <= df['res']) & (df['res'] <= bins_resolution[i + 1])]['Name'])

list_of_cluster['xcop'] = [x for x in XCOP_in_PSZ2 if x in list_of_cluster['reduced']]
