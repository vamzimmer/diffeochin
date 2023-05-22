import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

### General Imports
# import matplotlib.pyplot as plt
from glob import glob as glob
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from diffeochin.systemsetup import systemsetup_server as systemsetup
from diffeochin.utils import class_utils as cutils
import diffeochin.utils.eval_utils as eutils
import diffeochin.utils.do_analysis as do

info_file = systemsetup.INFO_FILE

n_components = 5
pairwise_embedding = 'kpca_rbf' 

# ORIGINAL SUBMISSION
subset_name = 'all_specimen'
methods = ['atlas_curve', 'atlas_surface', 'pairwise_curve', 'pairwise_surface']
data = ['cleaned_curve', 'simplified0.3', 'cleaned_curve', 'simplified0.3']
template_type = ['/template_5', '/template_5', '', '']

subset = ['European', 'African', 'Paranthropus', 'Australopithecus']

OUT_DIRs = ['{}/{}/{}-{}{}'.format(systemsetup.OUT_DIR, m.replace('_', '/'),subset_name, d, t) for m, d, t in zip(methods, data, template_type)]
print(OUT_DIRs)
saveit = True

do_embedding = True
do_taxon = False
do_gender = False
do_age = False
do_morph = False

for j, m in enumerate(methods):

    # output folders
    deformetrica_dir = OUT_DIRs[j] + '/atlasing' if 'atlas' in m else OUT_DIRs[j] + '/registrations'
    if not os.path.exists(deformetrica_dir):
        os.makedirs(deformetrica_dir)
    kpca_dir = OUT_DIRs[j] + '/kpca_{}'.format(n_components)
    if pairwise_embedding=='kpca_rbf':
        kpca_dir = OUT_DIRs[j] + '/kpca_rbf_{}'.format(n_components)
    mds_dir = OUT_DIRs[j] + '/mds_{}'.format(n_components)
    stat_dir = OUT_DIRs[j] + '/statistics_{}'.format(n_components)

    # read info file
    df = eutils.read_data(info_file, subset=subset, sheet=subset_name)

    '''
        EMBEDDING
    '''
    if do_embedding:
        if 'atlas' in m:
            # read momenta
            momenta_linearized = do.load_atlas_data(deformetrica_dir, df)

            # continue
            
            # perform kpca
            eig = do.perform_kpca(momenta_linearized, df, n_components)
            # save coordinates
            if not os.path.exists(kpca_dir):
                os.makedirs(kpca_dir)
            cutils.df_save_to_excel('{}/kpca.xlsx'.format(kpca_dir), df, '{}_PCs_ncomp{}'.format(subset_name,n_components))
            cutils.df_save_to_excel('{}/kpca.xlsx'.format(kpca_dir), eig, '{}_EV_ncomp{}'.format(subset_name,n_components))

        else:
            print(m)
            REG_DIR = '{}/pairwise/{}/all_specimen+-{}/registrations'.format(systemsetup.OUT_DIR, 'curve' if 'curve' in m else 'surface', 'cleaned_curve' if 'curve' in m else 'simplified0.3')
            D = do.load_distance_matrix(deformetrica_dir, REG_DIR, df, kernel='rbf' if pairwise_embedding=='kernel_rbf' else None)

            # continue

            if pairwise_embedding == 'mds':
                # perform mds
                do.perform_mds(D.drop(columns=['specimen']).to_numpy(), df, n_components)

                if not os.path.exists(mds_dir):
                    os.makedirs(mds_dir)
                cutils.df_save_to_excel('{}/mds.xlsx'.format(mds_dir), df, '{}_PCs_ncomp{}'.format(subset_name,n_components))
            else:
                # perform kpca
                eig = do.perform_kpca(D.drop(columns=['specimen']).to_numpy(), df, n_components, precomputed=True)
                # save coordinates
                if not os.path.exists(kpca_dir):
                    os.makedirs(kpca_dir)
                cutils.df_save_to_excel('{}/kpca.xlsx'.format(kpca_dir), df, '{}_PCs_ncomp{}'.format(subset_name,n_components))
                cutils.df_save_to_excel('{}/kpca.xlsx'.format(kpca_dir), eig, '{}_EV_ncomp{}'.format(subset_name,n_components))
    else:
        if 'atlas' in m or not pairwise_embedding=='mds':
            df = eutils.read_data('{}/kpca.xlsx'.format(kpca_dir), sheet='{}_PCs_ncomp{}'.format(subset_name,n_components))
        else:
            df = eutils.read_data('{}/mds.xlsx'.format(mds_dir), sheet='{}_PCs_ncomp{}'.format(subset_name,n_components))

    '''
        STATISTICAL ANALYSIS
    '''
    if 'pairwise' in m:
        stat_dir = stat_dir + '/' + pairwise_embedding

    if not os.path.exists(stat_dir):
        os.makedirs(stat_dir)
    stat_file = stat_dir + '/statistics.xlsx'

    subset = df['Group_names'].unique()
    subset_2 = df['Group_names_2'].unique()
    subset_2_used = ['European_male', 'European_female', 'African_male', 'African_female']

    print(subset)
    print(subset_2)

    #
    #   Taxon classification
    #
    if do_taxon:
        do.taxon_classification(df, subset, n_components, stat_file.replace('.xlsx', '_taxon.xlsx'), merge=True, showit=False)

    #
    #   Gender classification
    #
    if do_gender:
        do.gender_classification(df, subset[:2], subset_2_used, n_components, stat_file.replace('.xlsx', '_gender.xlsx'), showit=False)

    #
    #   Age regression
    #
    if do_age:
        do.age_regression(df, n_components, stat_file.replace('.xlsx', '_age_nz.xlsx'), showit=False, zscore=False,)


    #
    #   Morph regression
    #
    if do_morph:
        do.morph_regression(df, n_components, stat_file.replace('.xlsx', '_morph.xlsx'), showit=False)
        do.morph_regression(df, n_components, stat_file.replace('.xlsx', '_morph.xlsx'), showit=False, abs=True)
        