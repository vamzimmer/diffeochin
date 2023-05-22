import os
import copy
import deformetrica
import numpy as np
import pandas as pd
from ..utils import create_xml as cx
from ..utils import python_utils as putils


def pairwise_registration(mesh1, mesh2, output_dir, param_file):

    print("    Register " + os.path.splitext(os.path.basename(mesh1))[0] + " to " +
          os.path.splitext(os.path.basename(mesh2))[0])
    # out_dir = output_dir + "/" + os.path.splitext(os.path.basename(mesh1))[0] + "_to_" \
    #           + os.path.splitext(os.path.basename(mesh2))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file = output_dir + "/deformetrica.log"

    # create .xml files
    cx.optimization_parameters(output_dir + '/optimization_parameters.xml', 'GradientAscent', param_file)
    cx.model('Registration', output_dir + '/model.xml', param_file)
    cx.data_set(output_dir + '/data_set.xml', [mesh2], param_file)

    # launch deformetrica
    cmd = 'deformetrica estimate {}/model.xml {}/data_set.xml -p {}/optimization_parameters.xml --output={} -v INFO'\
        .format(output_dir, output_dir, output_dir, output_dir)
    print(cmd)
    os.system('%s 2>&1 | tee %s' % (cmd, log_file))

    print("===========================================")


def call_multiscale_pairwise_registration(mesh_moving, mesh_reference, output_dir, param_file):

    # read parameter file
    params = putils.read_parameter_file(param_file)

    # number of scales
    level = 1 if not isinstance(params['kernel_width_deformation'], list) else len(params['kernel_width_deformation'])

    # mesh_reference = mesh_files[0]
    # mesh_moving = mesh_files[60]
    reference = os.path.splitext(os.path.basename(mesh_reference))[0]
    moving = os.path.splitext(os.path.basename(mesh_moving))[0]

    odir = "{}/{}_to_{}".format(output_dir, moving, reference)
    if not os.path.exists(odir):
        os.makedirs(odir)
    if level > 1:
        # perform multiscale registration
        for lev in range(level):
            print("Level {}".format(lev))

            params_lev = copy.copy(params)
            for item in params_lev.items():
                if isinstance(item[1], list):
                    params_lev[item[0]] = item[1][lev]
            params_lev['template'] = mesh_moving
            if lev>0:
                params_lev['template'] = "{}/lev{}/DeterministicAtlas__Reconstruction__{}__subject_{}.vtk"\
                    .format(odir, lev-1, params_lev['object_id'], reference)

            pfile = "{}/parameters_{}.yml".format(odir, lev)
            putils.create_parameter_file(pfile, experiment=None, params=params_lev)
            pairwise_registration(mesh_moving, mesh_reference, "{}/lev{}".format(odir, lev), pfile)
    else:
        # only single scale registration
        params['template'] = mesh_moving
        pfile = "{}/parameters.yml".format(odir)
        putils.create_parameter_file(pfile, experiment=None, params=params)
        pairwise_registration(mesh_moving, mesh_reference, odir, pfile)


def run_pairwise_registrations(meshfiles, output_dir, param_file):
    # print(meshfiles)

    N = len(meshfiles)
    number_registrations = int(N * (N - 1) / 2)

    count = 1

    for i in range(0, N):

        print(" ")
        print("i=" + str(i) + ": target specimen " + os.path.splitext(os.path.basename(meshfiles[i]))[0])
        print(" ")

        for j in range(i+1,N):
            
            reference = os.path.splitext(os.path.basename(meshfiles[j]))[0]
            moving = os.path.splitext(os.path.basename(meshfiles[i]))[0]
            odir = "{}/{}_to_{}".format(output_dir, moving, reference)
            if os.path.exists('{}/DeterministicAtlas__EstimatedParameters__Momenta.txt'.format(odir)):
                print()
                print("Registration {} to {} already done.".format(moving, reference))
                print()
                continue
            # if not moving == 'fos_KW-7000' and not reference == 'fos_KW-7000':
            #     continue
            print("{}: {} of {}".format(os.path.splitext(os.path.basename(meshfiles[j]))[0], count, number_registrations))

            call_multiscale_pairwise_registration(meshfiles[i], meshfiles[j], output_dir, param_file)

            count = count + 1

    print("Done.")


def create_distance_matrix(data_dir, mesh_files, distance_file, N, kernel=None, gamma=0.01, overwrite=0):

    if N<len(mesh_files):
        file, ext = os.path.splitext(distance_file)
        distance_file = file + '_' + str(N) + ext
    if kernel is not None:
        distance_file = distance_file.replace('.xlsx', '_rbf.xlsx')

    if not os.path.isfile(distance_file) or overwrite:
        D = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, i):
                folder = data_dir + "/" + mesh_files[j] + "_to_" + mesh_files[i]
                momenta_file_in = folder + "/DeterministicAtlas__EstimatedParameters__Momenta.txt"

                momenta = np.loadtxt(momenta_file_in,skiprows=2)
                # momenta = momenta[1:, :]
                n_points = momenta.shape[0]

                dij = np.sum(np.sqrt(np.sum(np.square(momenta), axis=1))) / n_points
                if kernel is not None:
                    dij = np.exp(-gamma*dij)

                D[i, j] = dij
                D[j, i] = dij
        # np.savetxt(distance_file, D, fmt='%.5f', delimiter=',')
        df = pd.DataFrame(data=mesh_files)
        # insert content with column names
        for m in range(0, len(mesh_files)):
            df[mesh_files[m]] = D[:, m].tolist()

        df.to_excel(distance_file, sheet_name='distance_matrix')
    else:
        # D = np.loadtxt(distance_file, delimiter=',')
        df = pd.read_excel(distance_file, sheet_name='distance_matrix', engine='openpyxl')
        df = df.iloc[: , 1:]

    return df






