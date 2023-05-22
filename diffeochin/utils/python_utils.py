import os
import sys
import io
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook


def read_parameter_file(filename):
    # Read YAML file
    with open(filename, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def create_parameter_file(filename, experiment='pairwise-simplified0.3-all-roi1', params=None):

    if params is None:
        params = {}

        # default parameters
        params = {"experiment": 'default', "data_type": "SurfaceMesh", "object_id": "ID",
                  "kernel_type": "keops", "attachment": "Current",
                  "data_sigma": 0.1, "kernel_width_deformation": 5, "kernel_width_data": 5,
                  "initial_step": 1, "timepoints": 10, "tolerance": 1e-4, "freezetemplate": 'On',
                  "freezecp": 'Off', "maxIter": 200, 'gpu_mode': 'full'}

        if 'pairwise-simplified0.3' in experiment:  # experiment == 'pairwise-simplified0.3-all-roi1':
            params = {"experiment": experiment, "data_type": "SurfaceMesh", "object_id": "chin",
                      "kernel_type": "keops", "attachment": "Varifold",
                      "data_sigma": 0.1, "kernel_width_deformation": 5, "kernel_width_data": 8,
                      "initial_step": 20, "timepoints": 10, "tolerance": 1e-10, "freezetemplate": 'On',
                      "freezecp": 'Off', "maxIter": 10000, 'gpu_mode': 'full'}
        elif 'atlas-simplified0.3' in experiment:  # experiment == 'atlas-simplified0.3-all-roi1':
            params = {"experiment": experiment, "data_type": "SurfaceMesh", "object_id": "chin",
                      "kernel_type": "keops", "attachment": "Varifold",
                      "data_sigma": 0.1, "kernel_width_deformation": 3, "kernel_width_data": 8,
                      "initial_step": 20, "timepoints": 10, "tolerance": 1e-10, "freezetemplate": 'Off',
                      "freezecp": 'On', "maxIter": 10000, 'gpu_mode': 'full'}
        #
        # ROI2: Curves
        elif 'pairwise-curve' in experiment:  # experiment == 'pairwise-curve-all-roi2':
            params = {"experiment": experiment, "data_type": "Polyline", "object_id": "chin",
                      "kernel_type": "keops", "attachment": "Varifold",
                      "data_sigma": 0.1, "kernel_width_deformation": 3, "kernel_width_data": 8,
                      "initial_step": 1, "timepoints": 10, "tolerance": 1e-10, "freezetemplate": 'On',
                      "freezecp": 'Off', "maxIter": 1000, 'gpu_mode': 'full'}
        elif 'atlas-curve' in experiment:  # experiment == 'atlas-cleanedcurve-all-roi2':
            params = {"experiment": experiment, "data_type": "Polyline", "object_id": "chin",
                      "kernel_type": "keops", "attachment": "Varifold",
                      "data_sigma": 0.1, "kernel_width_deformation": 3, "kernel_width_data": 8,
                      "initial_step": 1, "timepoints": 10, "tolerance": 1e-5, "freezetemplate": 'Off',
                      "freezecp": 'On', "maxIter": 1000, 'gpu_mode': 'full'}

    # Write YAML file
    with io.open(filename, 'w', encoding='utf8') as outfile:
        yaml.dump(params, outfile, default_flow_style=False, allow_unicode=True)


def get_convergence_values(log_file, conv_file):
    with open(log_file) as read_file:
        content = read_file.readlines()

    loglike = [s for s in content if "Log-like" in s]
    values = [x.split()[3] + ' ' + x.split()[7] + ' ' + x.split()[11][:-1] for x in loglike]

    with open(conv_file, "w") as text_file:
        for v in values:
            text_file.write("{}\n".format(v))


def save_to_excel(outfile, data, column_names, sheet_name, row_names=None):
    if os.path.isfile(outfile):
        book = load_workbook(outfile)
        writer = pd.ExcelWriter(outfile, engine='openpyxl')
        writer.book = book

    # content
    if isinstance(data, np.ndarray):
        N, M = data.shape
    elif isinstance(data, list):
        M = len(data)
        N = 0
        for i in data:
            if len(i) > N:
                N = len(i)
    else:
        print("Data type not supported. 'data' is 'np.ndarray' or 'list'.")
        return None

    # create dataframe
    d = {'ids': range(0, N)}
    df = pd.DataFrame(data=d)

    if row_names is not None:
        df['names'] = row_names

    # insert content with column names
    for m in range(0, len(column_names)):
        if isinstance(data, np.ndarray):
            df[column_names[m]] = data[:, m].tolist()
        elif isinstance(data, list):
            df[column_names[m]] = data[m] + ['']*(N-len(data[m]))

    if os.path.isfile(outfile):
        df.to_excel(writer, sheet_name=sheet_name)
        writer.save()
        writer.close()
    else:
        df.to_excel(outfile, sheet_name=sheet_name)









