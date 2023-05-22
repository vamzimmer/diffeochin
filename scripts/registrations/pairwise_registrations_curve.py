"""
Created on Wed May 26 2021

@author: Veronika Zimmer

This function generates pairwise registrations with deformetrica 4.0.
For each surface pair, the .xml files are created automatically (chindef.utils.create_xml).
For N surfaces, N*(N-1)/2 registrations are performed.


"""

import os
import sys
# from matplotlib import pyplot as plt
# from glob import glob as glob
# import numpy as np
import pandas as pd
import yaml
import io

import chindef.utils as cutils
from chindef.systemsetup import systemsetup_kallisto as systemsetup
# from chindef.systemsetup import systemsetup as systemsetup
# from chindef.pairwise import call_deformetrica as cdef
from chindef.call_deformetrica import pairwise
import chindef.utils.python_utils as putils

info_file = systemsetup.INFO_FILE
sheet = 'all_specimen'
exp = 'ROI2-all-specimen'
experiment = 'pairwise-curve-all-roi2'

# sheet = 'all_specimen+'
# exp = 'ROI2-all-specimen+'
# experiment = 'pairwise-curve-all+-roi2'

data = 'cleaned_curve'
DATA1_DIR = "{}/{}".format(systemsetup.DATA_DIR, data)
OUT_DIR = '{}/pairwise/curve/{}-{}'.format(systemsetup.OUT_DIR, sheet, data)


def curve_matching():

    ##############################################################################################
    ##
    ##  Load specimen ids
    ##
    ##############################################################################################

    df = pd.read_excel(info_file, sheet_name=sheet) # sheet_name='ids_all')

    specimen = df['specimen']
    specimen = list(specimen)

    meshfiles = [DATA1_DIR+'/'+w+'.vtk' for w in specimen]
    N = len(meshfiles)

    print(N)
    # print(meshfiles)

    ##############################################################################################
    ##
    ##  Parameters
    ##
    ##############################################################################################

    out_dir = OUT_DIR + '/registrations'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    param_file = '{}/parameters.yaml'.format(out_dir)
    putils.create_parameter_file(param_file, experiment=experiment)

    ##############################################################################################
    ##
    ##  Launch pairwise registration
    ##
    ##############################################################################################
    pairwise.run_pairwise_registrations(meshfiles, out_dir, param_file)


def main():
    """
    main
    """
    curve_matching()


if __name__ == "__main__":
    main()
