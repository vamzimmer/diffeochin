import matplotlib.pyplot as plt
from glob import glob as glob
import numpy as np
import pandas as pd
import os
import matplotlib.cm as cm
from vtk import vtkPolyDataReader
from vtk import vtkPolyDataWriter


def read_data(info_file, subset=None, sheet='all_specimen'):

    '''
    Read file
    '''
    df = pd.read_excel(info_file, sheet_name=sheet,engine='openpyxl') # sheet_name='ids_all')
    # print(df.shape)
    if subset is not None:
        df = df.loc[df['Group_names'].isin(subset)]
        df = df.reset_index(drop=True)
    # print(df['Group_names'].unique())

    return df


def save_data(info_file, out_file, data_dir):

    '''
        Save PC coordinates for all experiments.
    '''

    subsets = [['European', 'African'], ['European', 'African', 'Paranthropus', 'Australopithecus']]
    subset_names = ['modern_humans', 'all_specimen']
    experiments = [['atlas', 'curve'], ['atlas', 'surface'], ['pairwise', 'curve'], ['pairwise', 'surface']]

    

    for sub, subn in zip(subsets, subset_names):

        # read data
        df = read_data(info_file, subset=sub)

        for exp in experiments:
            # read PC coordinated from kpca or mds

            var_met = ['PC', 'kpca', '/template_5'] if 'atlas' in exp else ['MDS', 'mds', '']
            var_roi = 'simplified0.3' if 'surface' in exp else 'cleaned_curve'

            direc = '{}/{}/{}/{}-{}{}/'.format(data_dir, exp[0], exp[1], subn, var_roi, var_met[2])
            
            df_pc = '{}/{}/{}.xlsx'



