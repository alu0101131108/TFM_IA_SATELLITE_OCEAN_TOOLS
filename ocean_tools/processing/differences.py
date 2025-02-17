# ocean_tools/processing/differences.py

import numpy as np
import matplotlib.pyplot as plt

def perform_difference_analysis(ds1_array, ds2_array, output_type='total'):
    n_diffs = len(ds1_array)
    diffs = []
    for i in range(n_diffs):
        if output_type == 'total':
            diffs.append(ds2_array[i] - ds1_array[i])
        elif output_type == 'latitudinal':
            diffs.append(ds2_array[i].mean(axis=1) - ds1_array[i].mean(axis=1))
        elif output_type == 'longitudinal':
            diffs.append(ds2_array[i].mean(axis=0) - ds1_array[i].mean(axis=0))
        elif output_type == 'total_average':
            diffs.append(ds2_array[i].mean() - ds1_array[i].mean())

    return diffs