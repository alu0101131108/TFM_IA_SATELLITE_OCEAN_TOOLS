# ocean_tools/processing/differences.py

import numpy as np
import matplotlib.pyplot as plt

def perform_difference_analysis(ds1_array, ds2_array, output_type='total'):
    """
    Computes the difference between two lists of numpy arrays based on the specified output type.

    This function assumes that ds1_array and ds2_array are lists (or array-like sequences)
    of equal length, where each element is a numpy array representing a spatial dataset.
    The difference is computed pairwise for each corresponding element in ds1_array and ds2_array.

    Parameters
    ----------
    * ds1_array : list
        * A list of numpy arrays representing the first set of data (e.g., historic data).
    * ds2_array : list
        * A list of numpy arrays representing the second set of data (e.g., recent data).
    * output_type : str, optional
        * The type of difference to compute. Options are:
            * 'total': returns the element-wise difference for each array (e.g., a 2D map),
            * 'latitudinal': returns the difference of the mean values along axis 1 (resulting in a 1D latitudinal profile),
            * 'longitudinal': returns the difference of the mean values along axis 0 (resulting in a 1D longitudinal profile),
            * 'total_average': returns a scalar representing the overall average difference.
        * Default is 'total'.

    Returns
    -------
    * list
        * A list of differences computed for each pair of arrays from ds1_array and ds2_array.
        * The content of each element depends on the output_type:
            * For 'total': each element is a numpy array (same shape as input arrays).
            * For 'latitudinal' and 'longitudinal': each element is a 1D numpy array.
            * For 'total_average': each element is a scalar.
    """
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