# ocean_tools/processing/pca.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.decomposition import IncrementalPCA
import pickle
import os

def prep_for_PCA(ds, variable_name):
    """
    Recibe un Dataset y retorna un arreglo 2D (time, lat*lon) 
    con valores normalizados (media 0, std 1), rellenando NaN con 0.
    """
    anom_prep = np.ma.masked_invalid(ds.variables[variable_name]).filled(0.)
    anom_prep = (anom_prep - anom_prep.mean()) / anom_prep.std()

    Ntime, Nlat, Nlon = anom_prep.shape
    anom_prep.shape = (Ntime, Nlat*Nlon)
    return anom_prep


def EOF_anomalies_analysis(anom_prep, n_components, store=False, store_dir="", file_name=""):
    """
    Perform an Incremental Principal Component Analysis (PCA) on a 2D array of anomalies (time, lat*lon).
    Parameters:
    anom_prep (numpy.ndarray): 2D array of anomalies with shape (time, lat*lon).
    n_components (int): Number of principal components to compute.
    store (bool, optional): If True, store the results to disk. Default is False.
    store_dir (str, optional): Directory where the results will be stored. Default is an empty string.
    file_name (str, optional): Base name for the stored files. Default is an empty string.
    Returns:
    tuple: A tuple containing:
        - LAM (numpy.ndarray): Explained variance of each principal component.
        - E (numpy.ndarray): Principal components (EOFs), with shape (lat*lon, n_components).
    Notes:
    - The function uses IncrementalPCA from scikit-learn to handle large datasets that do not fit into memory.
    - If `store` is True, the explained variance (LAM) and the principal components (E) are saved as pickle files in the specified directory.
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
    Grafica la fracción de varianza explicada por los primeros n_components autovalores
    con barras de error según la regla de North.
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
    Extrae los patrones espaciales (EOFs) y las series temporales (PCs).
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
    Identifica fechas de valor máximo y mínimo para la serie temporal asociada a cada PC.
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
