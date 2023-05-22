"""
Created on Wed May 26 2021

@author: Veronika Zimmer

This function constructs an atlas with deformetrica 4.0.
The .xml files are created automatically (chindef.utils.create_xml).
For N surfaces, N registrations are performed.


"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath('../..'))
import diffeochin.utils as cutils
from diffeochin.systemsetup import systemsetup_server as systemsetup
from diffeochin.call_deformetrica import atlas
import diffeochin.utils.python_utils as putils

info_file = systemsetup.INFO_FILE
sheet = 'all_specimen'
exp = 'ROI2-all-specimen'
experiment = 'atlas-curve'

data = 'cleaned_curve'
template_type = 'template_5'

DATA1_DIR = "{}/{}".format(systemsetup.DATA_DIR, data)
template = '{}/{}.vtk'.format(DATA1_DIR, template_type)
OUT_DIR = '{}/atlas/curve/{}-{}/{}'.format(systemsetup.OUT_DIR, sheet, data, template_type)


def atlas_construction():
    ##############################################################################################
    ##
    ##  Load specimen ids
    ##
    ##############################################################################################

    df = pd.read_excel(info_file, sheet_name=sheet)  # sheet_name='ids_all')

    specimen = df['specimen']
    specimen = list(specimen)

    meshfiles = [DATA1_DIR + '/' + w + '.vtk' for w in specimen]
    N = len(meshfiles)

    print(N)

    ##############################################################################################
    ##
    ##  Parameters and template
    ##
    ##############################################################################################

    out_dir = OUT_DIR + '/atlasing'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    param_file = '{}/parameters.yaml'.format(out_dir)
    putils.create_parameter_file(param_file, experiment=experiment)
    params = putils.read_parameter_file(param_file)
    params['template'] = template
    putils.create_parameter_file(param_file, experiment=None, params=params)

    ##############################################################################################
    ##
    ## Launch atlas construction
    ##
    ##############################################################################################

    log_file = atlas.atlas(meshfiles, out_dir, param_file)

    ##############################################################################################
    ##
    ## Check convergence
    ##
    ##############################################################################################

    conv_file = log_file.replace('deformetrica.log', 'convergence.txt')
    putils.get_convergence_values(log_file, conv_file)
    putils.plot_convergence(conv_file, out_dir)


def main():
    """
    main
    """
    atlas_construction()


if __name__ == "__main__":
    main()
