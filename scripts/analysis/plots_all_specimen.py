import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

### General Imports
# import matplotlib.pyplot as plt
from glob import glob as glob
import numpy as np
# import pandas as pd
import os
# import matplotlib.cm as cm
# from vtk import vtkPolyDataReader
# from vtk import vtkPolyDataWriter
# from numpy import linalg as LA


# from chindef.systemsetup import systemsetup_kallisto as systemsetup
from chindef.systemsetup import systemsetup_sinope as systemsetup
# from chindef.call_deformetrica import atlas
# from chindef.call_deformetrica import utils
# import chindef.utils.python_utils as putils
# import chindef.utils.plot_utils as pltutils
# import chindef.utils.class_utils as cutils
import chindef.utils.eval_utils as eutils
# import chindef.utils.stat_utils as sutils
import chindef.utils.do_analysis as do

info_file = systemsetup.INFO_FILE
subset_name = 'all_specimen'
subset_name = 'all_specimen+'
# subset_name = 'all_specimen_revision'
# subset_name = 'all_specimen+_revision'

revision = False
if 'revision' in subset_name:
    revision = True

subset = ['European', 'African', 'Paranthropus', 'Australopithecus']
if '+' in subset_name:
    subset = ['European', 'African', 'Paranthropus', 'Australopithecus', 'Early_Homo?']
n_components = 5
pairwise_embedding = 'kpca_rbf' #'mds', 'kpca'

# methods = ['atlas_curve', 'atlas_surface', 'pairwise_curve', 'pairwise_surface']
# data = ['cleaned_curve', 'simplified0.3', 'cleaned_curve', 'simplified0.3']
# template_type = ['/template_5', '/template_5', '', '']

methods = ['atlas_curve']
data = ['cleaned_curve']
methods = ['atlas_surface']
data = ['simplified0.3']
template_type = ['/template_5']
methods = ['pairwise_curve']
data = ['cleaned_curve']
methods = ['pairwise_surface']
data = ['simplified0.3']
template_type = ['']

OUT_DIRs = ['{}/{}/{}-{}{}'.format(systemsetup.OUT_DIR, m.replace('_', '/'),subset_name, d, t) for m, d, t in zip(methods, data, template_type)]
print(OUT_DIRs)

embedd_2d = True

for j, m in enumerate(methods):

    # if not j==1:
        # continue
    # if not 'pairwise' in m:
    #     continue
    print(m)
    print()

    # output folders
    kpca_dir = OUT_DIRs[j] + '/kpca_{}'.format(n_components)
    if pairwise_embedding=='kpca_rbf':
        kpca_dir = OUT_DIRs[j] + '/kpca_rbf_{}'.format(n_components)
    mds_dir = OUT_DIRs[j] + '/mds_{}'.format(n_components)
    stat_dir = OUT_DIRs[j] + '/statistics_{}'.format(n_components)
    if 'pairwise' in m:
        stat_dir += '/' + pairwise_embedding

    if 'atlas' in m or not pairwise_embedding=='mds':
        df = eutils.read_data('{}/kpca.xlsx'.format(kpca_dir), sheet='{}_PCs_ncomp{}'.format(subset_name,n_components))
        # read percentages explained by PC axis
        df_eig = eutils.read_data('{}/kpca.xlsx'.format(kpca_dir), sheet='{}_EV_ncomp{}'.format(subset_name,n_components))
        pc_var = np.zeros(n_components)
        pc_cum = list(df_eig['cum. variability (in %)'])
        pc_var[0] = pc_cum[0]
        for i in range(1,n_components):
            pc_var[i] = pc_cum[i] - pc_cum[i-1]
    else:
        df = eutils.read_data('{}/mds.xlsx'.format(mds_dir), sheet='{}_PCs_ncomp{}'.format(subset_name,n_components))

    # continue

    subset = df['Group_names'].unique()
    subset_2 = df['Group_names_2'].unique()

    if 'pairwise' in m and pairwise_embedding=='mds':
        pfile_2d = '{}/mds.png'.format(mds_dir) 
        pfile_2d_id = '{}/mds_id.png'.format(mds_dir) 
        pfile_3d = '{}/mds_3d.png'.format(mds_dir)
        gifs = [mds_dir+'/mds.gif', mds_dir+'/mds_gender.gif', '{}/mds_European.gif'.format(mds_dir), '{}/mds_African.gif'.format(mds_dir)]
    else:
        pfile_2d = '{}/kpca.png'.format(kpca_dir) 
        pfile_2d_id = '{}/kpca_id.png'.format(kpca_dir) 
        pfile_3d = '{}/kpca_3d.png'.format(kpca_dir)
        gifs = [kpca_dir+'/kpca.gif', kpca_dir+'/kpca_gender.gif', '{}/kpca_European.gif'.format(kpca_dir), '{}/kpca_African.gif'.format(kpca_dir)]
   
    axis_label = ['PC 1 ({:1.1f}%)'.format(pc_var[0]), 'PC 2 ({:1.1f}%)'.format(pc_var[1])]
    # do.plot_embeddings_2d(df, subset, subset_2, pfile_2d, axis_label=axis_label, revision=revision)
    # do.plot_embeddings_2d(df, subset, subset_2, pfile_2d_id, axis_label=axis_label, ids=True, revision=revision)

    # continue
    axis_label = ['PC 1 ({:1.1f}%)'.format(pc_var[0]), 'PC 2 ({:1.1f}%)'.format(pc_var[1]), 'PC 3 ({:1.1f}%)'.format(pc_var[2])]
    do.plot_embeddings_3d(df, subset, subset_2, pfile_3d, gifs, axis_label=axis_label, revision=revision)

    continue

    # select best correlating PCs
    variables = ['taxon', 'gender', 'age', 'morph']
    pcs = []
    for v in variables:
        stat_file = stat_dir + '/statistics_{}.xlsx'.format(v)
        mm = 'LogReg' if v=='taxon' or v=='gender' else 'LinReg'
        df_tax = eutils.read_data(stat_file, sheet='{}_ANOVA_ncomp{}'.format(mm,n_components))
        # print(df_tax)
        pc = np.argmin(df_tax['PR(>F)'])
        pcs.append(pc)
    print(pcs)

        # plot 2d embedding with most discriminating pc
    if 'pairwise' in m and pairwise_embedding=='mds':
            direc = mds_dir + '/most-discriminative-pc'
    else:
        direc = kpca_dir + '/most-discriminative-pc'
    if not os.path.exists(direc):
        os.makedirs(direc)
    for v, p in zip(variables, pcs):

        pfile_2d = '{}/{}_{}.png'.format(direc, pairwise_embedding, v)
        pfile_id_2d = '{}/{}_{}_id.png'.format(direc, pairwise_embedding, v)
        # print(pfile_2d)

        if not p:
            # if the most discriminant component is PC1, we plot PC1 vs. PC2 (again)
            print(v)
            axis_label = ['PC 1 ({:1.1f}%)'.format(pc_var[0]), 'PC 2 ({:1.1f}%)'.format(pc_var[1])]
            do.plot_embeddings_2d(df, subset, subset_2, pfile_2d, axis_label=axis_label, revision=revision)
            do.plot_embeddings_2d(df, subset, subset_2, pfile_id_2d, axis_label=axis_label, ids=True, revision=revision)
        else:
            # if the most discriminant component is NOT PC1, we plot PC1 vs. this PC
            print(v)
            axis_label = ['PC 1 ({:1.1f}%)'.format(pc_var[0]), 'PC {} ({:1.1f}%)'.format(p+1, pc_var[p])]
            do.plot_embeddings_2d(df, subset, subset_2, pfile_2d, plot_pcs=[1, p+1], axis_label=axis_label, revision=revision)
            do.plot_embeddings_2d(df, subset, subset_2, pfile_id_2d, plot_pcs=[1, p+1], axis_label=axis_label, ids=True, revision=revision)
        
    # continue

    if len(set([pcs[0], pcs[1]]))==2:
        # plot coordinates related to Taxon and Gender
        if 'pairwise' in m and pairwise_embedding=='mds':
            direc = mds_dir + '/taxon-gender'
        else:
            direc = kpca_dir + '/taxon-gender'
        if not os.path.exists(direc):
            os.makedirs(direc)
        pfile_2d = '{}/{}.png'.format(direc, pairwise_embedding)
        axis_label = ['PC {} ({:1.1f}%)'.format(pcs[0]+1, pc_var[pcs[0]]), 'PC {} ({:1.1f}%)'.format(pcs[1]+1, pc_var[pcs[1]])]
        do.plot_embeddings_2d(df, subset, subset_2, pfile_2d, plot_pcs=[pcs[0]+1, pcs[1]+1], axis_label=axis_label, revision=revision)

    if len(set([pcs[0], pcs[2]]))==2:
        # plot coordinates related to Taxon and Age
        if 'pairwise' in m and pairwise_embedding=='mds':
            direc = mds_dir + '/taxon-age'
        else:
            direc = kpca_dir + '/taxon-age'
        if not os.path.exists(direc):
            os.makedirs(direc)
        pfile_2d = '{}/{}.png'.format(direc, pairwise_embedding)
        axis_label = ['PC {} ({:1.1f}%)'.format(pcs[0]+1, pc_var[pcs[0]]), 'PC {} ({:1.1f}%)'.format(pcs[2]+1, pc_var[pcs[2]])]
        do.plot_embeddings_2d(df, subset, subset_2, pfile_2d, plot_pcs=[pcs[0]+1, pcs[2]+1], axis_label=axis_label, revision=revision)

    if len(set([pcs[0], pcs[3]]))==2:
        # plot coordinates related to Taxon and Morph
        if 'pairwise' in m and pairwise_embedding=='mds':
            direc = mds_dir + '/taxon-morph'
        else:
            direc = kpca_dir + '/taxon-morph'
        if not os.path.exists(direc):
            os.makedirs(direc)
        pfile_2d = '{}/{}.png'.format(direc, pairwise_embedding)
        axis_label = ['PC {} ({:1.1f}%)'.format(pcs[0]+1, pc_var[pcs[0]]), 'PC {} ({:1.1f}%)'.format(pcs[3]+1, pc_var[pcs[3]])]
        do.plot_embeddings_2d(df, subset, subset_2, pfile_2d, plot_pcs=[pcs[0]+1, pcs[3]+1], axis_label=axis_label, revision=revision)

    # continue

    if len(set([pcs[0], pcs[1], pcs[3]]))==3:
        if 'pairwise' in m and pairwise_embedding=='mds':
            direc = mds_dir + '/taxon-gender-morph'
        else:
            direc = kpca_dir + '/taxon-gender-morph'
        if not os.path.exists(direc):
            os.makedirs(direc)
        gifs = [direc+'/kpca.gif', direc+'/kpca_gender.gif', '{}/kpca_European.gif'.format(direc), '{}/kpca_African.gif'.format(direc)]
        axis_label = ['PC {} ({:1.1f}%)'.format(pcs[0]+1, pc_var[pcs[0]]), 'PC {} ({:1.1f}%)'.format(pcs[1]+1, pc_var[pcs[1]]), 'PC {} ({:1.1f}%)'.format(pcs[3]+1, pc_var[pcs[3]])]
        do.plot_embeddings_3d(df, subset, subset_2, pfile_3d, gifs, plot_pcs=[pcs[0]+1, pcs[1]+1, pcs[3]+1], axis_label=axis_label, revision=revision)

    
    print()
    print("Done ({})".format(m))
    print()