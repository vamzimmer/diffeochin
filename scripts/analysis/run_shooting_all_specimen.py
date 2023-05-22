import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

### General Imports
import matplotlib.pyplot as plt
from glob import glob as glob
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.cm as cm
# from vtk import vtkPolyDataReader
# from vtk import vtkPolyDataWriter
# from numpy import linalg as LA
# import pyvista as pv
# pv.set_plot_theme("document")

# from pyvista.utilities import xvfb
# xvfb.start_xvfb()
# 
# from IPython.display import Image

from chindef.systemsetup import systemsetup_kallisto as systemsetup
# from chindef.systemsetup import systemsetup as systemsetup
from chindef.call_deformetrica import atlas
from chindef.call_deformetrica import utils
import chindef.utils.python_utils as putils
import chindef.utils.plot_utils as pltutils
import chindef.utils.class_utils as cutils
import chindef.utils.eval_utils as eutils

info_file = systemsetup.INFO_FILE
subset_name = 'all_specimen'
subset_name = 'all_specimen_pop3'
subset_name = 'all_specimen_revision'

subset = ['European', 'African', 'Paranthropus', 'Australopithecus']
subsetX = ['Modern humans', 'Hominins']
n_components = 5
embedding = 'kpca_rbf'

# methods = ['atlas_curve', 'atlas_surface',]
# data = ['cleaned_curve', 'simplified0.3']
# methods = ['atlas_curve']
# data = ['cleaned_curve']
methods = ['atlas_surface']
data = ['simplified0.3']
template_type = '/template_5'

OUT_DIRs = ['{}/{}/{}-{}{}'.format(systemsetup.OUT_DIR, m.replace('_', '/'),subset_name, d, template_type) for m, d in zip(methods, data)]
print(OUT_DIRs)

saveit = True
do_vector = True
do_momentas = True
do_shooting = True
do_point_displacements = True
do_group_differences = True
do_diffs_on_template = True
do_cyclic_deformation = True

for j, m in enumerate(methods):
    # if j<1:
        # continue

    template = '{}/{}/{}.vtk'.format(systemsetup.DATA_DIR, data[j], template_type) 
    atlas_dir = '{}/atlasing'.format(OUT_DIRs[j])
    kpca_dir = '{}/{}_{}'.format(OUT_DIRs[j], embedding, n_components)
    shape_dir = '{}/shapes_{}'.format(OUT_DIRs[j], n_components)
    shooting_dir = '{}/shooting_{}'.format(OUT_DIRs[j], n_components)

    if not os.path.exists(shape_dir):
        os.makedirs(shape_dir)
    if not os.path.exists(shooting_dir):
        os.makedirs(shooting_dir)

    #
    #   Load info about specimen
    #
    df = eutils.read_data(kpca_dir+'/kpca.xlsx', sheet='{}_PCs_ncomp{}'.format(subset_name,n_components))

    subset = df['Group_names'].unique()
    subset_2 = df['Group_names_2'].unique()

    print(subset)
    print(subset_2)
    print(df.shape)
    # df.head(5)

    if do_vector:
        #
        #   Vector separating modern humans and fossils
        #
        group_means = df.groupby(['Group_names']).mean()
        if n_components==5:
            means = group_means[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].to_numpy()
        elif n_components==3:
            means = group_means[['PC1', 'PC2', 'PC3']].to_numpy()

        # separation goes from group1 (modern humans) to group2 (fossils)
        separation_vector = means[1,:] - means[0,:]
        print(separation_vector)

        # Plot
        g1 = df['Group_ids'].values < 2 # Modern humans
        g2 = df['Group_ids'].values > 1 # Hominins
        pfile = shooting_dir + '/separation_vector.png'
        X = df[['PC1', 'PC2', 'PC3']].to_numpy()
        limits = [np.min(X[:,0])-0.1, np.max(X[:,0])+0.1, np.min(X[:,1])-0.1, np.max(X[:,1])+0.1]
        pltutils.plot_vector(X, separation_vector, g1, g2, df['Group_ids'], [['European', 'African'], ['P.robustus', 'A. africanus']], pfile=pfile, limits=limits)

    # continue
    if do_momentas:
        #
        #   Projection of Moments
        #
        eig = eutils.read_data(kpca_dir+'/kpca.xlsx', sheet='{}_EV_ncomp{}'.format(subset_name,n_components))
        eigenvalues = eig['lambda'].to_numpy()
        alpha = eig['alpha'].to_numpy()
        eigenvectors = np.zeros((len(df), len(eigenvalues)))
        for d in range(len(eigenvalues)):
            eigenvectors[:,d] = np.fromstring(alpha[d][1:-1], sep=' ')
        f = open(atlas_dir + '/DeterministicAtlas__EstimatedParameters__Momenta.txt')
        first_line = f.readline().split(' ')
        number_of_subjects = int(first_line[0])
        number_of_controlpoints = int(first_line[1])
        dimension = int(first_line[2])
        f.close()
        momenta_linearised = np.loadtxt(atlas_dir+'/momentas.csv', delimiter=',')

        projection = np.matmul((eigenvalues * eigenvectors).transpose(), momenta_linearised)
        shoot = np.matmul(separation_vector, projection)
        shoot = shoot.reshape(number_of_controlpoints, dimension)

        if 'curve' in m:
            np.savetxt(shooting_dir + '/forward.txt',   .04 * shoot)
            np.savetxt(shooting_dir + '/backward.txt', -.04 * shoot)
        elif 'surface' in m:
            np.savetxt(shooting_dir + '/forward.txt',   .02 * shoot)
            np.savetxt(shooting_dir + '/backward.txt', -.02 * shoot)

    #
    #   Deformetrica shooting
    #
    if do_shooting:
        atlas.shooting(atlas_dir, shooting_dir)

    #
    #   Shooting output
    #
    # The shooting produces two sets of shapes in these directories:
    # - `.../shooting/forward`
    # - `.../shooting/backward`
    # They correspond to the mean shape `DeterministicAtlas__EstimatedParameters__Template` 
    # displaced towards the mean of group1 (`backward`) and the mean of group2 (`forward`)
    #
    # collect meshfiles
    meshfiles1 = glob('{}/forward/*.vtk'.format(shooting_dir))
    meshfiles2 = glob('{}/backward/*.vtk'.format(shooting_dir))

    meshfiles1.sort()
    meshfiles2.sort()

    list_order = [' '] * len(meshfiles1)
    for f in meshfiles1:
    #     idx = int(f[(f.index('__tp_')+len('__tp_')):f.index('.vtk')])
        idx = int(f[(f.index('__tp_')+len('__tp_')):f.index('__age')])
        list_order[idx] = f
    meshfiles1 = list_order

    list_order = [' '] * len(meshfiles2)
    for f in meshfiles2:
    #     idx = int(f[(f.index('__tp_')+len('__tp_')):f.index('.vtk')])
        idx = int(f[(f.index('__tp_')+len('__tp_')):f.index('__age')])
        list_order[idx] = f
    meshfiles2 = list_order

    shooting_df = pd.DataFrame()

    shooting_df['backward'] = [os.path.basename(meshfiles2[idx]) for idx in range(len(meshfiles2))]
    shooting_df['forward']  = [os.path.basename(meshfiles1[idx]) for idx in range(len(meshfiles1))]

    shooting_df

    #
    #   Group to template
    #   Template to group
    #
    if do_point_displacements:
        typ = ['point_displacements', 'surface_differences']
        folder = ['backward', 'forward']

        pfile_all = '{}/group_to_template.png'.format(shape_dir)
        pfile_all_2 = '{}/template_to_group.png'.format(shape_dir)
        meshes = []
        meshes_2 = []
        title = []
        for si, sub in enumerate(subsetX):

            # if si>0:
                # continue

            group_dir = '{}/{}_to_template'.format(shape_dir, sub)
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)

            mean = '{}/{}/Shooting__GeodesicFlow__chin__tp_10__age_1.00.vtk'.format(shooting_dir, folder[si])
            template = '{}/{}/Shooting__GeodesicFlow__chin__tp_0__age_0.00.vtk'.format(shooting_dir, folder[si])

            for ti, t in enumerate(typ):
                if ti>0 and 'curve' in m:
                    continue

                pfile = '{}/{}_to_template_{}.png'.format(group_dir, sub, t)
                mesh = '{}/{}_to_template_{}.vtk'.format(group_dir, sub, t)
                pfile2 = '{}/template_to_{}_{}.png'.format(group_dir, sub, t)
                mesh2 = '{}/template_to_{}_{}.vtk'.format(group_dir, sub, t)
                meshes.append(mesh)
                meshes_2.append(mesh2)
                title.append('{}-{}'.format(sub, t))


                atlas.compute_diff_to_template(mean, template, mesh, scalar=t)
                atlas.compute_diff_to_template(template, mean, mesh2, scalar=t)


    #
    #   Group to group differences
    #
    if do_group_differences:
        pfile_all = '{}/group_to_group.png'.format(shape_dir)
        typ = ['point_displacements', 'surface_differences']
        folder = ['backward', 'forward']

        group_dir = '{}/group_to_group'.format(shape_dir)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        mean1 = '{}/backward/Shooting__GeodesicFlow__chin__tp_10__age_1.00.vtk'.format(shooting_dir)
        mean2 = '{}/forward/Shooting__GeodesicFlow__chin__tp_10__age_1.00.vtk'.format(shooting_dir)
        mesh0 = '{}/group0-to-group1-point_displacements.vtk'.format(group_dir)
        atlas.compute_diff_to_template(mean1, mean2, mesh0, scalar='point_displacements')

        mesh1 = '{}/group1-to-group0-point_displacements.vtk'.format(group_dir)
        atlas.compute_diff_to_template(mean2, mean1, mesh1, scalar='point_displacements')

        if 'surface' in m:
            mesh2 = '{}/group0-to-group1-surface_differences.vtk'.format(group_dir)
            atlas.compute_diff_to_template(mean1, mean2, mesh2, scalar='surface_differences')

            mesh3 = '{}/group1-to-group0-surface_differences.vtk'.format(group_dir)
            atlas.compute_diff_to_template(mean2, mean1, mesh3, scalar='surface_differences')


    #
    #   Point differences on template
    #
    if do_diffs_on_template:
        pfile_all = '{}/group_to_group_on_template.png'.format(shape_dir)
        typ = ['point_displacements', 'surface_differences']
        folder = ['backward', 'forward']

        group_dir = '{}/on_template'.format(shape_dir)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        mean1 = '{}/backward/Shooting__GeodesicFlow__chin__tp_10__age_1.00.vtk'.format(shooting_dir)
        mean2 = '{}/forward/Shooting__GeodesicFlow__chin__tp_10__age_1.00.vtk'.format(shooting_dir)

        mesh0 = '{}/group0-to-group1-on-template-point_displacements.vtk'.format(group_dir)
        atlas.compute_groupdiff_on_template([mean1, mean2], template, mesh0, scalar='point_displacements')

        mesh1 = '{}/group1-to-group0-on-template-point_displacements.vtk'.format(group_dir)
        atlas.compute_groupdiff_on_template([mean1, mean2], template, mesh1, scalar='point_displacements')

        if 'surface' in m:
            mesh2 = '{}/group0-to-group1-on-template-surface_differences.vtk'.format(group_dir)
            atlas.compute_groupdiff_on_template([mean1, mean2], template, mesh2, scalar='surface_differences')

            mesh3 = '{}/group1-to-group0-on-template-surface_differences.vtk'.format(group_dir)
            atlas.compute_groupdiff_on_template([mean1, mean2], template, mesh3, scalar='surface_differences')


            

    # 
    #   Cyclic deformation between groups plus volume change
    #
    # We can re-order the shooting output shapes them in order to produce a list of shapes corresponding to a cyclic motion going from mean_g1 to mean_g2 and back.
    #
    # We can calculate the local change of volume between group 1 and group 2.
    # This can be derived by calculating the change of surface area from the mean shape (of the entire population) and each of the shape in the sequence.
    #
    # -----
    #
    # Note: The resulting measure is the change in surface area, therefore in $mm^2$. 
    # You can derive an **estimation** of the local volume change with the following equation:
    # $V = e^{i\pi} + 1$
    
    #
    #   Flow
    #
    if do_cyclic_deformation:

        # store everything in folder shape_dir/group_to_group
        group_dir = '{}/group_to_group'.format(shape_dir)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        gif = group_dir+'/group_to_group.gif'

        atlas.create_shooting_combination(shooting_dir, meshfiles1, meshfiles2, '{}/combined'.format(shooting_dir), scalar=None)

        # point displacements
        typ = 'point_displacements'
        gif = group_dir+'/group_to_group_{}.gif'.format(typ)

        atlas.create_shooting_combination(shooting_dir, meshfiles1, meshfiles2, '{}/combined_{}'.format(shooting_dir, typ), scalar=typ)
            
        if 'surface' in m:
            typ = 'surface_differences'
            gif = group_dir+'/group_to_group_{}.gif'.format(typ)

            atlas.create_shooting_combination(shooting_dir, meshfiles1, meshfiles2, '{}/combined_{}'.format(shooting_dir, typ), scalar=typ)
