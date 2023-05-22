import os
import copy
from PIL.Image import new
import deformetrica
from ..utils import create_xml as cx
from ..utils import python_utils as putils
import numpy as np

### VTK imports
import vtk
from vtk import vtkPolyDataReader
from vtk import vtkPolyDataWriter
from vtk import vtkUnstructuredGridReader
from vtk import vtkUnstructuredGridWriter
from vtk import vtkPolyData
from vtk import vtkPoints
from vtk import vtkCellArray
from vtk import vtkProcrustesAlignmentFilter as Procrustes
from vtk import vtkMultiBlockDataGroupFilter as GroupFilter
from vtk import vtkLandmarkTransform
from vtk import vtkTransformPolyDataFilter as TransformFilter
from vtk.util import vtkConstants
from vtk import vtkIdList
from vtk import vtkIdTypeArray
from vtk import vtkTriangle
from vtk import vtkFloatArray
from vtk import vtkTetra
from vtk import vtkMath


def atlas(mesh_files, output_dir, param_file):

    log_file = output_dir + "/deformetrica.log"

    # create xml.files
    cx.optimization_parameters(output_dir + '/optimization_parameters.xml', 'GradientAscent', param_file)
    cx.model('DeterministicAtlas', output_dir + '/model.xml', param_file)
    cx.data_set(output_dir + '/data_set.xml', mesh_files, param_file)

    # launch deformetrica
    cmd = 'deformetrica estimate {}/model.xml {}/data_set.xml -p {}/optimization_parameters.xml --output={} -v INFO' \
    .format(output_dir, output_dir, output_dir, output_dir)
    print(cmd)
    os.system('%s 2>&1 | tee %s' % (cmd, log_file))

    print("Done.")

    return log_file


def shooting(def_dir, out_dir):
    cx.optimization_parameters_shooting(out_dir + '/optimization_parameters.xml')
    cx.data_set_shooting(out_dir + '/data_set.xml', def_dir)
    cx.model_shooting(out_dir + '/model_forward.xml', def_dir, out_dir + '/forward.txt')
    cx.model_shooting(out_dir + '/model_backward.xml', def_dir, out_dir + '/backward.txt')

    direc = os.getcwd()
    os.chdir(out_dir)

    cmd = 'deformetrica compute ' + out_dir + '/model_forward.xml'
    print(cmd)
    os.system(cmd)
    os.system('mv output forward')

    cmd = 'deformetrica compute ' + out_dir + '/model_backward.xml'
    print(cmd)
    os.system(cmd)
    os.system('mv output backward')

    os.chdir(direc)


def cyclic_combination(forward_files, backward_files, destination):
    import shutil
    import os
    assert (len(forward_files) == len(backward_files))
    N = 4 * len(forward_files)
    combined_list = list(reversed(backward_files))
    combined_list.extend(forward_files)
    combined_list.extend(list(reversed(forward_files)))
    combined_list.extend(backward_files)

    for idx, f in enumerate(combined_list):
        dest = os.path.join(destination, 'separation_{:03d}.vtk'.format(idx))
        shutil.copy(f, dest)


def compute_cell_surfaces(polydata):
    pt_ids = vtkIdList()
    tr = vtkTriangle()
    areas = vtkFloatArray()
    areas.SetNumberOfComponents(1)
    areas.SetNumberOfTuples(polydata.GetNumberOfPolys())
    areas.SetName('Area')

    for idx in range(polydata.GetNumberOfPolys()):
        polydata.GetCellPoints(idx, pt_ids)
        pts = []
        for pt_ids_id in range(pt_ids.GetNumberOfIds()):
            pts.append(polydata.GetPoint(pt_ids.GetId(pt_ids_id)))
        assert (len(pts) == 3)
        v = tr.TriangleArea(pts[0], pts[1], pts[2])
        areas.SetTuple1(idx, v)

    polydata.GetCellData().SetScalars(areas)


def compute_cell_surfaces_differences(in1, in2, out):
    pt_ids = vtkIdList()
    tr = vtkTriangle()
    areas = vtkFloatArray()
    areas.SetNumberOfComponents(1)
    areas.SetNumberOfTuples(in1.GetNumberOfPolys())
    areas.SetName('Volume Change (in %)')

    for idx in range(in1.GetNumberOfPolys()):
        in1.GetCellPoints(idx, pt_ids)
        pts = []
        for pt_ids_id in range(pt_ids.GetNumberOfIds()):
            pts.append(in1.GetPoint(pt_ids.GetId(pt_ids_id)))
        assert (len(pts) == 3)
        v1 = tr.TriangleArea(pts[0], pts[1], pts[2])
        in2.GetCellPoints(idx, pt_ids)
        pts = []
        for pt_ids_id in range(pt_ids.GetNumberOfIds()):
            pts.append(in2.GetPoint(pt_ids.GetId(pt_ids_id)))
        assert (len(pts) == 3)
        v2 = tr.TriangleArea(pts[0], pts[1], pts[2])
        areas.SetTuple1(idx, 100. * 2. * (v1 - v2) / (v1 + v2))

    out.GetCellData().SetScalars(areas)


def compute_cell_volume_differences(in1, in2, out):
    pt_ids = vtkIdList()
    tr = vtkTetra()
    areas = vtkFloatArray()
    areas.SetNumberOfComponents(1)
    areas.SetNumberOfTuples(in1.GetNumberOfCells())
    areas.SetName('Volume Change (in %)')

    for idx in range(in1.GetNumberOfCells()):
        in1.GetCellPoints(idx, pt_ids)
        pts = []
        for pt_ids_id in range(pt_ids.GetNumberOfIds()):
            pts.append(in1.GetPoint(pt_ids.GetId(pt_ids_id)))
        if (len(pts) != 4):
            continue

        v1 = tr.ComputeVolume(pts[0], pts[1], pts[2], pts[3])
        in2.GetCellPoints(idx, pt_ids)
        pts = []
        for pt_ids_id in range(pt_ids.GetNumberOfIds()):
            pts.append(in2.GetPoint(pt_ids.GetId(pt_ids_id)))
        assert (len(pts) == 4)
        v2 = tr.ComputeVolume(pts[0], pts[1], pts[2], pts[3])
        areas.SetTuple1(idx, 100. * 2. * (v1 - v2) / (v1 + v2))

    out.GetCellData().SetScalars(areas)


def compute_point_displacements(in1, in2, out):
    displacements = vtkFloatArray()
    displacements.SetNumberOfComponents(1)

    # My change to make the code work for curves (else statement)
    # @Veronika, 15.6.2021
    if in1.GetNumberOfPolys() > 0:
        displacements.SetNumberOfTuples(in1.GetNumberOfPolys())
    else:
        displacements.SetNumberOfTuples(in1.GetNumberOfPoints())
    displacements.SetName('Displacements (in mm)')

    for idx in range(in1.GetNumberOfPoints()):
        pt1 = in1.GetPoint(idx)
        pt2 = in2.GetPoint(idx)

        displacements.SetTuple1(idx, np.sqrt(vtkMath.Distance2BetweenPoints(pt1, pt2)))

    out.GetPointData().SetScalars(displacements)


def create_shooting_combination(shooting_dir, meshfiles1, meshfiles2, out_dir, scalar=None, template=None):

    # if not os.path.exists(shooting_dir + '/combined1'):
    #     os.mkdir(shooting_dir + '/combined1')
    # if not os.path.exists(shooting_dir + '/combined2'):
    #     os.mkdir(shooting_dir + '/combined2')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # without volume change
    # cyclic_combination(meshfiles1, meshfiles2, shooting_dir + '/combined1')

    if template is None:
        meshfile_c = '{}/forward/Shooting__GeodesicFlow__chin__tp_10__age_1.00.vtk'.format(shooting_dir)
    else:
        meshfile_c = template
    # meshfile_c = '{}/backward/Shooting__GeodesicFlow__chin__tp_0__age_0.00.vtk'.format(shooting_dir)
    # meshfile_c = '{}/forward/Shooting__GeodesicFlow__chin__tp_0__age_0.00.vtk'.format(shooting_dir)


    reader = vtkPolyDataReader()
    reader.SetFileName(meshfile_c)
    reader.Update()
    mean = reader.GetOutput()

    # meshes = meshfiles1 + meshfiles2
    meshes = [meshfiles1, meshfiles2]

    # type=None                 :   Without volume change
    # type='point_displacements :   Compute point displacements   
    # print(scalar)  
    if scalar is not None:

        mesh_new = [[],[]]
        for i in range(len(meshes)):


            if 'forward' in meshes[i][0]:
                change_dir = '{}/forward_{}'.format(shooting_dir, scalar)
            else:
                change_dir = '{}/backward_{}'.format(shooting_dir, scalar)
            if not os.path.exists(change_dir):
                os.makedirs(change_dir)

            for idx, mfile in enumerate(meshes[i]):
                ### read mesh
                reader1 = vtkPolyDataReader()
                reader1.SetFileName(mfile)
                reader1.Update()
                m = reader1.GetOutput()

                if scalar=='point_displacements':
                    # print('point')
                    compute_point_displacements(m, mean, m)
                elif scalar=='volume_differences':
                    # print('volume')
                    compute_cell_volume_differences(m, mean, m)
                elif scalar=='surface_differences':
                    # print('surface')
                    compute_cell_surfaces_differences(m, mean, m)

                # meshes[i][idx] = os.path.join(change_dir, os.path.basename(mfile))
                ofile = os.path.join(change_dir, os.path.basename(mfile))
                mesh_new[i].append(ofile)
                
                ### write mesh
                writer = vtkPolyDataWriter()
                writer.SetInputData(m)
                writer.SetFileName(ofile)
                writer.Update()
    
        # print(meshes[0][:5])
        cyclic_combination(mesh_new[0], mesh_new[1], out_dir)
    else:
        cyclic_combination(meshes[0], meshes[1], out_dir)



def compute_diff_to_template(mesh_file, template_file,out_file, scalar='point_displacements'):

    reader = vtkPolyDataReader()
    reader.SetFileName(template_file)
    reader.Update()
    template = reader.GetOutput()

    ### read mesh
    readerM = vtkPolyDataReader()
    readerM.SetFileName(mesh_file)
    readerM.Update()
    m = readerM.GetOutput()

    # compute_point_displacements(m, template, m)
    if scalar=='point_displacements':
        compute_point_displacements(m, template, m)
    elif scalar=='volume_differences':
        compute_cell_volume_differences(m, template, m)
    elif scalar=='surface_differences':
        compute_cell_surfaces_differences(m, template, m)

    ### write mesh
    writer = vtkPolyDataWriter()
    writer.SetInputData(m)
    writer.SetFileName(out_file)
    writer.Update()


def compute_groupdiff_on_template(mesh_files, template_file, out_file, scalar='point_displacements'):

    reader = vtkPolyDataReader()
    reader.SetFileName(template_file)
    reader.Update()
    template = reader.GetOutput()

    ### read mesh
    meshes = []
    for i in range(2):
        readerM = vtkPolyDataReader()
        readerM.SetFileName(mesh_files[i])
        readerM.Update()
        meshes.append(readerM.GetOutput())

    # compute_point_displacements(m, template, m)
    if scalar=='point_displacements':
        compute_point_displacements(meshes[0], meshes[1], template)
    elif scalar=='volume_differences':
        compute_cell_volume_differences(meshes[0], meshes[1], template)
    elif scalar=='surface_differences':
        compute_cell_surfaces_differences(meshes[0], meshes[1], template)

    ### write mesh
    writer = vtkPolyDataWriter()
    writer.SetInputData(template)
    writer.SetFileName(out_file)
    writer.Update()






