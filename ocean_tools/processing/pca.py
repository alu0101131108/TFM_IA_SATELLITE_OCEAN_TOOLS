# ocean_tools/processing/pca.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.decomposition import IncrementalPCA
import pickle
import os

def prep_for_PCA(ds, variable_name):
    """
    Prepares a dataset for PCA by normalizing the variable and reshaping it into a 2D array.

    The function converts the input variable (assumed to have dimensions (time, lat, lon))
    into a 2D array of shape (time, lat*lon) after normalizing the data to have zero mean and unit standard deviation.
    Any NaN values are replaced with 0.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable.
    variable_name : str
        Name of the variable to process.

    Returns
    -------
    np.ndarray
        2D numpy array with shape (time, lat*lon) containing the normalized data.
    """
    anom_prep = np.ma.masked_invalid(ds.variables[variable_name]).filled(0.)
    anom_prep = (anom_prep - anom_prep.mean()) / anom_prep.std()

    Ntime, Nlat, Nlon = anom_prep.shape
    anom_prep.shape = (Ntime, Nlat*Nlon)
    return anom_prep


def EOF_anomalies_analysis(anom_prep, n_components, store=False, store_dir="", file_name=""):
    """
    Performs Incremental Principal Component Analysis (PCA) on a 2D anomaly array.

    The function uses scikit-learn's IncrementalPCA to compute EOFs (Empirical Orthogonal Functions)
    from a 2D array of anomalies with shape (time, lat*lon). It returns the explained variance and the EOFs
    (principal components) as a tuple. Optionally, the results can be stored as pickle files.

    Parameters
    ----------
    anom_prep : np.ndarray
        2D array of anomalies with shape (time, lat*lon), normalized (zero mean, unit std).
    n_components : int
        Number of principal components to compute.
    store : bool, optional
        If True, the results (explained variance and EOFs) are saved to disk. Default is False.
    store_dir : str, optional
        Directory where the results will be stored if store is True.
    file_name : str, optional
        Base name for the stored files if store is True.

    Returns
    -------
    tuple
        A tuple (LAM, E) where:
          - LAM is a numpy.ndarray containing the explained variance of each component.
          - E is a numpy.ndarray of EOFs with shape (lat*lon, n_components).
    """
    ipca = IncrementalPCA(n_components=n_components, batch_size=100)
    ipca.fit_transform(anom_prep)

    LAM = ipca.explained_variance_
    E = ipca.components_.T

    if store:
        file_path = os.path.join(store_dir, f'{file_name}_LAM.pkl')
        base, ext = os.path.splitext(file_path)
        counter = 1
        new_file_path = file_path
        while os.path.exists(new_file_path):
            new_file_path = f"{base}_{counter}{ext}"
            counter += 1
        with open(new_file_path, 'wb') as f:
            pickle.dump(LAM, f)
        
        file_path = os.path.join(store_dir, f'{file_name}_E.pkl')
        base, ext = os.path.splitext(file_path)
        counter = 1
        new_file_path = file_path
        while os.path.exists(new_file_path):
            new_file_path = f"{base}_{counter}{ext}"
            counter += 1
        with open(new_file_path, 'wb') as f:
            pickle.dump(E, f)

    return LAM, E


def plot_eigenvalues_explained_variance(LAM, E, n_components, title='Fraction of Variance Explained'):
    """
    Plots the fraction of variance explained by the first n_components principal components (EOFs).

    The function displays a line plot of the normalized explained variance with error bars
    computed according to North's rule of thumb.

    Parameters
    ----------
    LAM : np.ndarray
        Array of explained variances from the PCA.
    E : np.ndarray
        Array of principal components (EOFs) with shape (lat*lon, n_components).
    n_components : int
        Number of components to plot.
    title : str, optional
        Title for the plot. Default is 'Fraction of Variance Explained'.

    Returns
    -------
    None
    """
    pc_ts = E[:, 0]
    pc_ts_std = (pc_ts - pc_ts.mean()) / pc_ts.std()

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, n_components + 1), LAM[:n_components] / LAM.sum(),
             '.-', color='gray', linewidth=2)

    Nstar = len(pc_ts_std)
    eb = LAM[:n_components] / LAM.sum() * np.sqrt(2./float(Nstar))
    plt.errorbar(np.arange(1, n_components + 1),
                 LAM[:n_components] / LAM.sum(),
                 yerr=eb/2, xerr=None, linewidth=1, color='gray')

    plt.title(title, fontsize=16)
    plt.xlabel('EOFs')
    plt.show()


def get_patterns_and_ts(E, n_patterns, nlat, nlon, anom_prep_var):
    """
    Extracts spatial patterns (EOFs) and temporal series (PCs) from the PCA results.

    For each of the first n_patterns EOFs, the function reshapes the EOF into a 2D array (nlat, nlon)
    representing the spatial pattern and computes the corresponding time series (PC) by projecting the 
    preprocessed anomaly data onto the EOF. The time series is normalized (zero mean, unit std).

    Parameters
    ----------
    E : np.ndarray
        EOFs (principal components) from PCA with shape (lat*lon, n_components).
    n_patterns : int
        Number of patterns (EOFs) to extract.
    nlat : int
        Number of latitude points.
    nlon : int
        Number of longitude points.
    anom_prep_var : np.ndarray
        The 2D anomaly array (time, lat*lon) used for PCA.

    Returns
    -------
    tuple
        A tuple (patterns, time_series) where:
          - patterns is a list of 2D arrays (each of shape (nlat, nlon)) representing spatial patterns.
          - time_series is a list of 1D arrays representing the corresponding temporal series (PCs).
    """
    patterns = []
    time_series = []

    for i in range(n_patterns):
        pat_2d = np.reshape(np.real(E[:, i]), (nlat, nlon))
        patterns.append(pat_2d)

        ts_i = np.dot(anom_prep_var, np.real(E[:, i]))
        ts_i = (ts_i - np.mean(ts_i)) / np.std(ts_i)
        time_series.append(ts_i)

    return patterns, time_series


def get_pattern_ts_max_min(ds_anom, time_series, variable_name, verbose=False):
    """
    Identifies the dates corresponding to the maximum and minimum values of each time series.

    For each time series (associated with an EOF), the function finds the indices of the maximum and minimum 
    values, retrieves the corresponding dates from ds_anom, and extracts the patterns (fields) corresponding 
    to those dates.

    Parameters
    ----------
    ds_anom : xr.Dataset
        Dataset containing the variable and the time coordinate.
    time_series : list
        List of 1D numpy arrays representing temporal series (PCs) for each EOF.
    variable_name : str
        Name of the variable in ds_anom.
    verbose : bool, optional
        If True, prints the dates corresponding to the maximum and minimum values for each EOF.
        Default is False.

    Returns
    -------
    list
        A list of lists, where each inner list contains two elements: [max_pattern, min_pattern],
        corresponding to the patterns (fields) on the dates of maximum and minimum values.
    """
    maxmins = []
    for i, ts in enumerate(time_series):
        max_i = np.argmax(ts)
        min_i = np.argmin(ts)

        if verbose:
            max_date = ds_anom.variables['time'][max_i]
            min_date = ds_anom.variables['time'][min_i]
            print(f"EOF {i+1} => Max date: {max_date}, Min date: {min_date}")

        min_pattern = ds_anom.variables[variable_name][min_i]
        max_pattern = ds_anom.variables[variable_name][max_i]
        maxmins.append([max_pattern, min_pattern])
    return maxmins
