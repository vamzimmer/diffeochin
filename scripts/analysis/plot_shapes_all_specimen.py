import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING) 

### General Imports
import pandas as pd
import os
import sys

import pyvista as pv
pv.set_plot_theme("document")

sys.path.insert(0, os.path.abspath('../..'))
from diffeochin.systemsetup import systemsetup_server as systemsetup
from diffeochin.utils import plot_utils as pltutils

info_file = systemsetup.INFO_FILE
subset_name = 'all_specimen'
subset = ['European', 'African', 'Paranthropus', 'Australopithecus', 'Early_Homo?']
n_components = 5
embedding = 'kpca_rbf'

# methods = ['atlas_curve', 'atlas_surface',]
# data = ['cleaned_curve', 'simplified0.3']
methods = ['atlas_curve']
data = ['cleaned_curve']
# methods = ['atlas_surface']
# data = ['simplified0.3']
template_type = '/template_5'

OUT_DIRs = ['{}/{}/{}-{}{}'.format(systemsetup.OUT_DIR, m.replace('_', '/'),subset_name, d, template_type) for m, d in zip(methods, data)]
print(OUT_DIRs)

camera_position = (34.16, 98.64, -25.20) # (30.96, 117.13, -42.55) # (20.18, 124.50, -25.51)
camera_focalpoint = (47.25, 56.75, 7.40,)
camera_viewup = (-0.97, -0.19, 0.14) # (-0.98, -0.17, 0.12) # (-0.93, -0.27, 0.20)
camera = [camera_position, camera_focalpoint, camera_viewup]

saveit = True
plot_means = False
plot_point_displacements = False
plot_group_differences = False
plot_diffs_on_template = False
plot_flow = True

for j, m in enumerate(methods):
    # if j>0:
        # continue
    # if j<1:
        # continue

    template = '{}/{}/{}.vtk'.format(systemsetup.DATA_DIR, data, template_type) 
    atlas_dir = '{}/atlasing'.format(OUT_DIRs[j])
    shape_dir = '{}/shapes_{}'.format(OUT_DIRs[j], n_components)
    shooting_dir = '{}/shooting_{}'.format(OUT_DIRs[j], n_components)
    
    #
    #   Template and mean shapes
    #
    template = '{}/DeterministicAtlas__EstimatedParameters__Template_chin.vtk'.format(atlas_dir)
    if plot_means:
        pfile = '{}/template_groupmeans.png'.format(shape_dir)
        mean0 = '{}/backward/Shooting__GeodesicFlow__chin__tp_10__age_1.00.vtk'.format(shooting_dir)
        mean1 = '{}/forward/Shooting__GeodesicFlow__chin__tp_10__age_1.00.vtk'.format(shooting_dir)

        # pltutils.plot_shapes(template, subplots=(1,1), titles='Template', colors='black', camera=camera, line_width=5)
        if 'curve' in m:
            pltutils.plot_shapes([[template, template], [mean0, mean1]], subplots=(1,2), titles=['Template', 'Mean groups'], colors=['dimgrey', 'darkgrey'], line_width=7, is2d=True, pfile=pfile, window_size=[1600, 500])
        elif 'surface' in m:
            pltutils.plot_shapes([template, mean0, mean1], subplots=(1,3), titles=['Template', 'Mean group 0', 'Mean group 1'], 
                        colors=['white', 'white', 'white'], camera=camera, pfile=pfile, window_size=[1600, 500])


    #
    #   Group to template
    #   Template to group
    #
    if plot_point_displacements:

        typ = ['point_displacements', 'surface_differences']
        folder = ['backward', 'forward']

        pfile_all = '{}/group_to_template.png'.format(shape_dir)
        pfile_all_2 = '{}/template_to_group.png'.format(shape_dir)
        meshes = []
        meshes_2 = []
        title = []
        subsetX = ['Modern humans', 'Hominins']
        for si, sub in enumerate(subsetX):

            # if si>0:
                # continue

            group_dir = '{}/{}_to_template'.format(shape_dir, sub)

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

        if 'curve' in m:
            pltutils.plot_shapes([[template, meshes[0]],[template, meshes[1]]], subplots=(1, 2), titles=subsetX, cmap='coolwarm', camera=None, pfile=pfile_all, line_width=7, is2d=True)
            pltutils.plot_shapes([[meshes_2[0]],[meshes_2[1]]], subplots=(1, 2), titles=subsetX, cmap='coolwarm', camera=None, pfile=pfile_all_2, line_width=7, is2d=True)
        elif 'surface' in m:
            pltutils.plot_shapes(meshes, subplots=(2, 2), titles=title, cmap='coolwarm', camera=camera, pfile=pfile_all)
            pltutils.plot_shapes(meshes_2, subplots=(2, 2), titles=title, cmap='coolwarm', camera=camera, pfile=pfile_all_2)

    #
    #   Group to group differences
    #
    if plot_group_differences:
        pfile_all = '{}/group_to_group.png'.format(shape_dir)

        group_dir = '{}/group_to_group'.format(shape_dir)
        subsetX = ['Modern humans', 'Hominins']

        mesh0 = '{}/group0-to-group1-point_displacements.vtk'.format(group_dir)
        mesh1 = '{}/group1-to-group0-point_displacements.vtk'.format(group_dir)

        if 'curve' in m:
            titles = ['Point differences: Group0 to group1', 'Point differences: Group1 to group0']
            pltutils.plot_shapes([[mesh0], [mesh1]], subplots=(1,2), cmap='coolwarm', titles=titles, camera=None, pfile=pfile_all, show=True, line_width=7, is2d=True)
        elif 'surface' in m:
            mesh3 = '{}/group1-to-group0-surface_differences.vtk'.format(group_dir)

            titles = ['Point differences: Group0 to group1', 'Point differences: Group1 to group0', 'Surface differences: Group0 to group1', 'Surface differences: Group1 to group0']
            pltutils.plot_shapes([[mesh0], [mesh1], [mesh2], [mesh3]], subplots=(2,2), cmap='coolwarm', titles=titles, camera=camera, pfile=pfile_all, show=True)



    #
    #   Group to group on template
    #
    if plot_diffs_on_template:
        pfile_all = '{}/group_to_group_on_template.png'.format(shape_dir)
        group_dir = '{}/on_template'.format(shape_dir)

        mesh0 = '{}/group0-to-group1-on-template-point_displacements.vtk'.format(group_dir)
        mesh1 = '{}/group1-to-group0-on-template-point_displacements.vtk'.format(group_dir)

        if 'curve' in m:
            titles = ['Point differences on template: Group0 to group1', 'Point differences on template: Group1 to group0']
            pltutils.plot_shapes([[mesh0], [mesh1]], subplots=(1,2), cmap='coolwarm', titles=titles, camera=None, line_width=7, is2d=True, 
                                pfile=pfile_all, show=True)
        elif 'surface' in m:
            mesh2 = '{}/group0-to-group1-on-template-surface_differences.vtk'.format(group_dir)
            mesh3 = '{}/group1-to-group0-on-template-surface_differences.vtk'.format(group_dir)

            titles = ['Point differences on template: Group0 to group1', 'Point differences on template: Group1 to group0', 'Surface differences on template: Group0 to group1', 'Surface differences on template: Group1 to group0']
            pltutils.plot_shapes([[mesh0], [mesh1], [mesh2], [mesh3]], subplots=(2,2), cmap='coolwarm', titles=titles, camera=camera, 
                                    pfile=pfile_all, show=True)

    #
    #   Cyclic flow
    #
    camera_position = (34.16, 98.64, -25.20) # (30.96, 117.13, -42.55) # (20.18, 124.50, -25.51)
    camera_focalpoint = (47.25, 56.75, 7.40,)
    camera_viewup = (-0.97, -0.19, 0.14) # (-0.98, -0.17, 0.12) # (-0.93, -0.27, 0.20)
    camera = [camera_position, camera_focalpoint, camera_viewup]
    if plot_flow:
        group_dir = '{}/group_to_group'.format(shape_dir)
        gif = group_dir+'/group_to_group.gif'
        if 'curve' in m:
            pltutils.create_gif('{}/combined'.format(shooting_dir), gif, camera_position=None, line_width=7, is2d=True)
        elif 'surface' in m:
            pltutils.create_gif('{}/combined'.format(shooting_dir), gif, camera_position=camera)

        # point displacements
        typ = 'point_displacements'
        gif = group_dir+'/group_to_group_{}.gif'.format(typ)
        if 'curve' in m:
            pltutils.create_gif('{}/combined_{}'.format(shooting_dir, typ), gif, camera_position=None, line_width=7, is2d=True, clim=(0,4.71))
        elif 'surface' in m:
            pltutils.create_gif('{}/combined_{}'.format(shooting_dir, typ), gif, camera_position=camera, clim=(0,6.0))

        # surface differences
        typ = 'surface_differences'
        gif = group_dir+'/group_to_group_{}.gif'.format(typ)

        if 'surface' in m:
            pltutils.create_gif('{}/combined_{}'.format(shooting_dir, typ), gif, camera_position=camera, clim=(-69.2,43.0))

