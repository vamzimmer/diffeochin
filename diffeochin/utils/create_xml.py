import os

from diffeochin.utils import python_utils as putils


def optimization_parameters(filename, method, param_file):
    params = putils.read_parameter_file(param_file)
    print(params)

    with open(filename, "w") as text_file:
        text_file.write('<?xml version=\"1.0\"?>\n')
        text_file.write('<optimization-parameters>\n')
        text_file.write('\t<optimization-method-type>%s</optimization-method-type>\n' % method)
        text_file.write('\t<initial-step-size>%1.5f</initial-step-size>\n' % params['initial_step'])
        text_file.write('\t<freeze-template>%s</freeze-template>\n' % params['freezetemplate'])
        text_file.write('\t<freeze-control-points>%s</freeze-control-points>\n' % params['freezecp'])
        text_file.write('\t<gpu-mode>%s</gpu-mode>\n' % params['gpu_mode'])
        text_file.write('\t<convergence-tolerance>%1.10f</convergence-tolerance>\n' % params['tolerance'])
        text_file.write('\t<max-iterations>%d</max-iterations>\n' % params['maxIter'])
        text_file.write('</optimization-parameters>')


def optimization_parameters_shooting(filename):
    with open(filename, "w") as text_file:
        text_file.write('<?xml version=\"1.0\"?>\n')
        text_file.write('<optimization-parameters>\n')
        text_file.write('</optimization-parameters>')


def model(type, filename, param_file):
    params = putils.read_parameter_file(param_file)

    with open(filename, "w") as text_file:
        text_file.write('<?xml version=\"1.0\"?>\n')
        text_file.write('<model>\n')
        text_file.write('\t<model-type>%s</model-type>\n' % type)
        dim = 3 if params['data_type'] == 'SurfaceMesh' else 2
        text_file.write('\t<dimension>%s</dimension>\n' % dim)
        text_file.write('\t<template>\n')
        text_file.write('\t\t<object id=\"%s\">\n' % params['object_id'])
        text_file.write('\t\t\t<deformable-object-type>%s</deformable-object-type>\n' % params['data_type'])
        text_file.write('\t\t\t<attachment-type>%s</attachment-type>\n' % params['attachment'])
        # text_file.write('\t\t\t<attachment-type>Varifold</attachment-type>\n')# % data_sigma)
        text_file.write('\t\t\t<noise-std>%1.1f</noise-std>\n' % params['data_sigma'])
        text_file.write('\t\t\t<kernel-type>%s</kernel-type>\n' % params['kernel_type'])
        text_file.write('\t\t\t<kernel-width>%1.1f</kernel-width>\n' % params['kernel_width_data'])
        text_file.write('\t\t\t<filename>%s</filename>\n' % params['template'])
        text_file.write('\t\t</object>\n')
        text_file.write('\t</template>\n')
        text_file.write('\t<deformation-parameters>\n')
        text_file.write('\t\t<kernel-width>%1.1f</kernel-width>\n' % params['kernel_width_deformation'])
        text_file.write('\t\t<kernel-type>%s</kernel-type>\n' % params['kernel_type'])
        text_file.write('\t\t<number-of-timepoints>%d</number-of-timepoints>\n' % params['timepoints'])
        text_file.write('\t</deformation-parameters>\n')
        text_file.write('</model>\n')


def model_shooting(filename, atlas_dir, mom_vals):
    atlas_model = atlas_dir + '/model.xml'
    cp_pos = atlas_dir + "/DeterministicAtlas__EstimatedParameters__ControlPoints.txt"

    # reading from atlas parameter file
    with open(atlas_model) as read_file:
        model = read_file.readlines()

    obj = [s for s in model if "object id" in s][0]
    object_id = obj[obj.find('"') + 1:obj.rfind('"')]
    ds = [s for s in model if "noise-std" in s][0]
    data_sigma = ds[15:ds.rfind('<')]
    kt = [s for s in model if "kernel-type" in s][0]
    kernel_type = kt[16:kt.rfind('<')]
    kw = [s for s in model if "kernel-width" in s][0]
    kernel_width_data = kw[17:kw.rfind('<')]
    kw = [s for s in model if "kernel-width" in s][1]
    kernel_width_def = kw[16:kw.rfind('<')]
    kt = [s for s in model if "kernel-type" in s][1]
    kernel_type2 = kt[15:kt.rfind('<')]
    tp = [s for s in model if "timepoints" in s][0]
    timepoints = tp[24:tp.rfind('<')]
    dt = [s for s in model if "deformable-object-type" in s][0]
    data_type = dt[27:dt.rfind('<')]

    template = atlas_dir + '/DeterministicAtlas__EstimatedParameters__Template_' + object_id + '.vtk'

    with open(filename, "w") as text_file:
        text_file.write('<?xml version=\"1.0\"?>\n')
        text_file.write('<model>\n')
        text_file.write('\t<model-type>Shooting</model-type>\n')
        text_file.write('\t<initial-control-points>%s</initial-control-points>\n' % cp_pos)
        text_file.write('\t<initial-momenta>%s</initial-momenta>\n' % mom_vals)
        text_file.write('\t<template>\n')
        text_file.write('\t\t<object id=\"%s\">\n' % object_id)
        text_file.write('\t\t\t<deformable-object-type>%s</deformable-object-type>\n' % data_type)
        text_file.write('\t\t\t<noise-std>%s</noise-std>\n' % data_sigma)
        text_file.write('\t\t\t<kernel-type>%s</kernel-type>\n' % kernel_type)
        text_file.write('\t\t\t<kernel-width>%s</kernel-width>\n' % kernel_width_data)
        #         text_file.write('\t\t\t<filename>no filename for shooting</filename>\n')
        text_file.write('\t\t\t<filename>%s</filename>\n' % template)
        text_file.write('\t\t</object>\n')
        text_file.write('\t</template>\n')
        text_file.write('\t<deformation-parameters>\n')
        text_file.write('\t\t<kernel-width>%s</kernel-width>\n' % kernel_width_def)
        text_file.write('\t\t<kernel-type>%s</kernel-type>\n' % kernel_type2)
        text_file.write('\t\t<number-of-timepoints>%s</number-of-timepoints>\n' % timepoints)
        text_file.write('\t</deformation-parameters>\n')
        text_file.write('</model>\n')


def data_set(filename, files, param_file):
    params = putils.read_parameter_file(param_file)

    with open(filename, "w") as text_file:
        text_file.write('<?xml version=\"1.0\"?>\n')
        text_file.write('<data-set>\n')
        for s in range(0, len(files)):
            text_file.write('\t<subject id=\"%s\">\n' % os.path.splitext(os.path.basename(files[s]))[0])
            text_file.write('\t\t<visit id=\"%s\">\n' % params['experiment'])
            text_file.write('\t\t\t<filename object_id=\"%s\">%s</filename>\n' % (params['object_id'], files[s]))
            text_file.write('\t\t</visit>\n')
            text_file.write('\t</subject>\n')
        text_file.write('</data-set>\n')


def data_set_shooting(filename, atlas_dir):

    atlas_data = atlas_dir + '/data_set.xml'
    # reading from atlas parameter files
    with open(atlas_data) as read_file:
        data = read_file.readlines()

    visit = [s for s in data if "visit id" in s][0]
    visit_id = visit[visit.find('"') + 1:visit.rfind('"')]

    obj = [s for s in data if "object_id" in s][0]
    object_id = obj[obj.find('"') + 1:obj.rfind('"')]

    template = atlas_dir + '/DeterministicAtlas__EstimatedParameters__Template_' + object_id + '.vtk'

    with open(filename, "w") as text_file:
        text_file.write('<?xml version=\"1.0\"?>\n')
        text_file.write('<data-set>\n')
        text_file.write('\t<subject id=\"template\">\n')
        text_file.write('\t\t<visit id=\"%s\">\n' % visit_id)
        text_file.write('\t\t\t<filename object_id=\"%s\">%s</filename>\n' % (object_id, template))
        text_file.write('\t\t</visit>\n')
        text_file.write('\t</subject>\n')
        text_file.write('</data-set>\n')





