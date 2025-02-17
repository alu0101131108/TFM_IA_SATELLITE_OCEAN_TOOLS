#### Estructura de directorios de ocean_tools
# # ocean_tools/
# # ├── __init__.py
# # ├── config/
# # │   ├── __init__.py
# # │   └── defaults.py
# # ├── data_handling/
# # │   ├── __init__.py
# # │   ├── download.py
# # │   └── availability.py
# # ├── io/
# # │   ├── __init__.py
# # │   ├── readers.py
# # │   └── writers.py
# # ├── processing/
# # │   ├── __init__.py
# # │   ├── anomalies.py
# # │   ├── data_prep.py
# # │   ├── merges.py
# # │   ├── pca.py
# # │   └── regression.py
# # └──── visualization/
# #     ├── __init__.py
# #     ├── maps.py
# #     └── reports.py


# ocean_tools/config/defaults.py

REGIONS = {
    'cclme': {
        'lat': [5, 40],
        'lon': [-30, -5]
    },
    'weak_permanent_uw': {
        'lat': [26, 35],
        'lon': [-30, -5]
    },
    'permanent_uw': {
        'lat': [21, 26],
        'lon': [-30, -12]
    },
    'mauritania_senegalese_uw': {
        'lat': [12, 19],
        'lon': [-30, -15]
    }
}

VARIABLES = {
    'sst': 'Sea Surface Temperature',
    'chlor_a': 'Chlorophyll-a'
}

CLIM_DEFAULTS = {
    'sst': [0, 35],
    'chlor_a': [0.01, 20]
}

# ocean_tools/data_handling/availability.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def data_availability_analysis(urls_directory: str, monthly: bool = False, plot: bool = True) -> None:
    """
    Lee un archivo de texto con URLs, agrupa sus fechas y analiza la disponibilidad
    (diaria o mensual). Opcionalmente grafica el recuento de archivos por mes.
    """
    with open(urls_directory, "r") as file:
        file_urls = file.read()

    url_list = list(filter(None, map(str.strip, file_urls.split("\n"))))
    file_names = [url.split("/")[-1] for url in url_list]

    dates = [file_name.split(".")[1][:8] for file_name in file_names]
    df = pd.DataFrame(dates, columns=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df.sort_values(by="Date", inplace=True)

    df_monthly = df.groupby(df["Date"].dt.to_period("M")).agg(
        files_available=pd.NamedAgg(column="Date", aggfunc="count")
    )

    date_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq='MS')
    df_monthly = df_monthly.reindex(date_range.to_period('M'))
    df_monthly["files_available"] = df_monthly["files_available"].fillna(0)

    df_monthly["total_days"] = df_monthly.index.map(lambda x: 1 if monthly else x.daysinmonth)
    df_monthly["missing_days"] = df_monthly["total_days"] - df_monthly["files_available"]

    total_files = df_monthly['files_available'].sum()
    ideal_files = df_monthly['total_days'].sum()

    print(f"Found {total_files} files out of idealy {ideal_files} files.")
    print(f"Completion Rate: {total_files / ideal_files * 100:.2f}%")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

    if plot:
        df_monthly["files_available"].plot(kind="bar", figsize=(35, 5), color="green", edgecolor="black")
        df_monthly["missing_days"].plot(kind="bar", figsize=(35, 5), color="red", edgecolor="black", bottom=df_monthly["files_available"])
        plt.title(f"Availability over {os.path.basename(urls_directory)}")
        plt.xlabel("Year-Month")
        plt.ylabel("Number of files")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


# ocean_tools/data_handling/download.py

import os
import time
import shutil
import webbrowser
import requests
from pathlib import Path

def wait_for_and_move_file(file_name: str, destination_path: str, timeout: int = 60, verbose: bool = False) -> None:
    """
    Espera a que aparezca un archivo en la carpeta de descargas 
    y lo mueve a la ruta especificada.
    """
    downloads_dir = str(Path.home() / "Downloads")
    file_path = os.path.join(downloads_dir, file_name)
    destination_file = Path(destination_path)

    start_time = time.time()

    while not os.path.exists(file_path):
        elapsed_time = int(time.time() - start_time)
        if elapsed_time > 0 and int(elapsed_time) % 5 == 0 and verbose:
            print(f"Waiting {int(elapsed_time)} seconds for {file_name}...")
        time.sleep(1)
        if elapsed_time > timeout:
            raise FileNotFoundError(
                f"File '{file_name}' not found in {downloads_dir} after {timeout} seconds."
            )

    destination_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(file_path, destination_path)
    if verbose:
        print(f"File moved to {destination_path}")


def download_image_from_url(image_url: str, destination_path: str, verbose: bool = False) -> None:
    """
    Descarga un archivo de imagen (PNG) desde una URL y lo guarda en la ruta indicada.
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        if verbose:
            print(f"Image downloaded successfully to {destination_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")
        raise


def download_file_using_browser_and_move(in_url: str, destination_path: str, timeout: int = 60, verbose: bool = False) -> None:
    """
    Abre una URL en el navegador por defecto para descargar el archivo 
    y luego lo mueve a la carpeta de destino. 
    Si el archivo es .png, se descarga directamente.
    """
    file_name = in_url.split("/")[-1]
    output_path = os.path.join(destination_path, file_name)

    if file_name.lower().endswith(".png"):
        download_image_from_url(in_url, output_path, verbose)
    else:
        if verbose:
            print(f"Opening the URL in your default browser: {in_url}")
        webbrowser.open(in_url, new=2, autoraise=False)
        wait_for_and_move_file(file_name, output_path, timeout, verbose)


def bulk_download_files(file_urls: str, destination_path: str, max_files: int = 0, file_timeout: int = 60, verbose: bool = False) -> None:
    """
    Descarga múltiples archivos abriendo sus URLs en el navegador (o directamente si .png)
    y los mueve a la carpeta de destino.
    """
    url_list = list(filter(None, map(str.strip, file_urls.split("\n"))))
    n_total_files = len(url_list)

    # No se descargan archivos ya presentes en la carpeta de destino
    destination_files = os.listdir(destination_path)
    url_list = [url for url in url_list if url.split("/")[-1] not in destination_files]
    
    n_pending_files = len(url_list)
    n_skippable_files = n_total_files - n_pending_files

    if max_files > 0 and max_files < n_pending_files:
        url_list = url_list[:max_files]

    n_download_files = len(url_list)
    
    print(f"Total: {n_total_files} | Existing: {n_skippable_files} | Pending: {n_pending_files} | Downloading: {n_download_files}")
    start_time = time.time()

    effectively_downloaded = 0
    for i, url in enumerate(url_list, start=1):
        try:
            download_file_using_browser_and_move(url, destination_path, file_timeout, verbose)
            effectively_downloaded += 1
        except FileNotFoundError as e:
            print(f"Error downloading file: {e}")

        if verbose or i % 10 == 0 or i == n_download_files:
            print(f"Processed: {i} | Downloaded {effectively_downloaded} | Run time: {int(time.time() - start_time)}s")

    print(f"{effectively_downloaded} files downloaded successfully to {destination_path}.")

    if effectively_downloaded < n_download_files:
        print(f"{n_download_files - effectively_downloaded} files failed to download.")


# ocean_tools/io/readers.py

import xarray as xr

def get_xarray_from_file(file_path):
    """
    Carga un archivo NetCDF y devuelve el Dataset de xarray resultante.
    """
    ds = xr.open_dataset(file_path)
    return ds


# ocean_tools/io/writers.py
import xarray as xr
import os

def save_ndarray_as_netcdf_to_file(ref_ds, ndarray, var_name, file_path):
    """
    Save an ndarray as a NetCDF file at the specified path.

    Parameters:
    -----------
    ref_ds : xarray.Dataset
        Reference dataset containing latitude, longitude, and time coordinates.
    ndarray : np.ndarray
        The data array to save.
    var_name : str
        The name of the variable to save in the NetCDF file.
    file_path : str
        The output NetCDF file path.
    """
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path

    while os.path.exists(new_file_path):
        new_file_path = f"{base}_{counter}{ext}"
        counter += 1

    if len(ndarray.shape) == 3:
        ds_xarray = xr.DataArray(ndarray, dims=['time', 'lat', 'lon'], coords={'time': ref_ds.time, 'lat': ref_ds.lat, 'lon': ref_ds.lon}, name=var_name)
    elif len(ndarray.shape) == 2:
        ds_xarray = xr.DataArray(ndarray, dims=['lat', 'lon'], coords={'lat': ref_ds.lat, 'lon': ref_ds.lon}, name=var_name)
    else:
        raise ValueError("The dataset must have dimensions (lat, lon) or (time, lat, lon).")
    ds_xarray.to_netcdf(new_file_path)


# ocean_tools/processing/anomalies.py

import xarray as xr

def get_deseasonalized_anomaly_ds(ds, variable_name):
    """
    Lee un dataset y calcula la anomalía (deseasonalizada) para 'variable_name',
    restando a cada mes su promedio mensual respectivo.
    """
    month_target_means = []
    for i in range(1, 13):
        month_target = ds.sel(time=ds['time.month'] == i)
        month_target_mean = month_target[variable_name].mean(axis=0)
        month_target_means.append(month_target_mean)

    # Recorre cada "time" y resta la media mensual correspondiente
    for i, date in enumerate(ds.variables['time'].values):
        month = int(date.astype('datetime64[M]').astype(int) % 12 + 1)
        ds[variable_name][i] = ds[variable_name][i] - month_target_means[month - 1]

    return ds


# ocean_tools/processing/data_prep.py

import numpy as np
from .anomalies import get_deseasonalized_anomaly_ds
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def prepare_dataset_for_analysis(ds, variable_name, use_anomalies=False, anomaly_transform='none'):
    """
    1) Se eliminan los datos que no pertenezcan al océano, como ríos o lagos.
    2) Si use_anomalies=True, se calcula la anomalía deseasonalizada,
       y luego, según anomaly_transform:
         - 'positive': conserva sólo valores > 0 (filtra negativos)
         - 'negative': conserva sólo valores < 0 (filtra positivos)
         - 'square': eleva al cuadrado
         - 'abs': valor absoluto
         - 'none': no aplica filtrado/transformación posterior
    """
    
    # Generar máscara oceano/continente.
    proj = {'projection': ccrs.PlateCarree()}
    fig, ax = plt.subplots(figsize=(len(ds.lon)/100, len(ds.lat)/100), dpi=100, subplot_kw=proj)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    ax.set_frame_on(False)
    ax.set_extent([ds.lon[0], ds.lon[-1], ds.lat[0], ds.lat[-1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='black')
    fig.canvas.draw()
    mask = fig.canvas.buffer_rgba()
    ncols, nrows = fig.canvas.get_width_height()
    plt.close(fig)
    mask = np.frombuffer(mask, dtype=np.uint8).reshape(nrows, ncols, 4)
    mask = mask[:, :, :3]  # Keep only the RGB channels
    mask = mask.mean(axis=2)  # Merge RGB into Grayscale
    mask = (mask > 128).astype(float)  # Convert to binary mask with a threshold

    # Aplicar máscara para eliminar datos fuera del océano.
    ds[variable_name] = ds[variable_name].where(mask == 1)
    
    if not use_anomalies:
        return ds  # Devuelve dataset crudo

    # Calcula anomalías deseasonalizadas
    ds_anom = get_deseasonalized_anomaly_ds(ds, variable_name)

    # Aplica transformaciones
    if anomaly_transform == 'positive':
        ds_anom[variable_name] = ds_anom[variable_name].where(ds_anom[variable_name] > 0)
    elif anomaly_transform == 'negative':
        ds_anom[variable_name] = ds_anom[variable_name].where(ds_anom[variable_name] < 0)
    elif anomaly_transform == 'square':
        ds_anom[variable_name] = ds_anom[variable_name] ** 2
    elif anomaly_transform == 'abs':
        ds_anom[variable_name] = np.abs(ds_anom[variable_name])

    return ds_anom

def time_aggregator(ds, variable_name, mode='full', agg_type='mean'):
    """
    Realiza una agregación temporal (media, desviación estándar, etc.) para un dataset basado en el modo especificado.
    
    Parámetros:
    ----------
    ds : xarray.Dataset
        El dataset de entrada que contiene los datos.
    variable_name : str
        El nombre de la variable para la que se realiza la agregación.
    mode : str
        El modo de cálculo: 'full', 'seasonal' o 'monthly'.
    agg_type : str
        El tipo de agregación: 'mean', 'std', etc. (cualquier método soportado por xarray).

    Retorno:
    -------
    Si mode == 'full': xarray.DataArray con la agregación global de la variable.
    Si mode == 'seasonal': dict con 4 xarray.DataArray (invierno, primavera, verano, otoño).
    Si mode == 'monthly': dict con 12 xarray.DataArray, uno para cada mes.
    """
    if not hasattr(ds[variable_name], agg_type):
        raise ValueError(f"Tipo de agregación '{agg_type}' no soportado por xarray.")

    if mode == 'full':
        # Agregación total a través de la dimensión temporal
        return getattr(ds[variable_name], agg_type)(dim='time')

    elif mode == 'seasonal':
        # Diccionario para almacenar las agregaciones estacionales
        seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11]
        }
        seasonal_aggregates = {}
        for season, months in seasons.items():
            season_ds = ds.sel(time=ds['time.month'].isin(months))
            seasonal_aggregates[season] = getattr(season_ds[variable_name], agg_type)(dim='time')
        return seasonal_aggregates

    elif mode == 'monthly':
        # Diccionario para almacenar las agregaciones mensuales
        monthly_aggregates = {}
        for month in range(1, 13):
            month_ds = ds.sel(time=ds['time.month'] == month)
            monthly_aggregates[month] = getattr(month_ds[variable_name], agg_type)(dim='time')
        return monthly_aggregates

    else:
        raise ValueError(f"Modo '{mode}' no reconocido. Usa 'full', 'seasonal' o 'monthly'.")


# ocean_tools/processing/merges.py

import os
import xarray as xr

def merge_variable_days_preprocess(dataset_dir, file, lat, lon, variable_name):
    """
    Abre un archivo NetCDF (expandiendo la dimensión 'time'), 
    filtra la región y mantiene sólo 'variable_name'.
    """
    import datetime as dt

    file_path = os.path.join(dataset_dir, file)
    ds = xr.open_dataset(file_path).expand_dims('time')
    date = file.split('.')[1][:8]
    ds['time'] = [dt.date(int(date[0:4]), int(date[4:6]), int(date[6:8]))]

    ds = ds.where(
        (ds.lat >= lat[0]) & (ds.lat <= lat[1]) &
        (ds.lon >= lon[0]) & (ds.lon <= lon[1]),
        drop=True
    )

    ds = ds[[variable_name]]

    return ds


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

# ocean_tools/processing/regression.py

import numpy as np
from sklearn import linear_model

def raster_series_regression_2d_slopes(ds, variable_name, step_name="Step"):
    """
    Realiza una regresión lineal en cada punto (lat, lon) de un Dataset,
    calculando la pendiente de la serie temporal de 'variable_name'.
    """
    time_dim = ds.variables['time'].astype('int64')
    lat = ds.variables['lat']
    lon = ds.variables['lon']

    reggresor = linear_model.LinearRegression()
    slope = np.full((len(lat), len(lon)), np.nan)

    progress = 0
    for i in range(len(lat)):
        for j in range(len(lon)):
            y = ds[variable_name][:, i, j].values
            X = time_dim.values.reshape(-1, 1)

            # Eliminar NaNs
            mask = ~np.isnan(y)
            X_f = X[mask]
            y_f = y[mask]

            if len(y_f) > 0:
                model = reggresor.fit(X_f, y_f)
                slope[i, j] = model.coef_.item()

        new_progress = int((i / len(lat)) * 100)
        if new_progress > progress:
            progress = new_progress
            print(f"{step_name} - {progress}%")

    return slope

from ..io.writers import save_ndarray_as_netcdf_to_file
def get_regression_slopes(ds, variable_name, mode='full', step_name="Step", store=False, store_dir="", file_name=""):
    """
    Calcula las pendientes de regresión para un dataset basado en el modo especificado.
    
    Parámetros:
    ----------
    ds : xarray.Dataset
        El dataset de entrada que contiene los datos.
    variable_name : str
        El nombre de la variable para la que se calculan las pendientes.
    mode : str
        El modo de cálculo: 'full', 'seasonal' o 'monthly'.
    step_name : str
        El nombre del paso para mostrar el progreso.
    store : bool
        Si True, guarda los resultados en un archivo NetCDF.
    store_dir : str
        La ruta donde se almacenará el archivo NetCDF.
    file_name : str
        El nombre del archivo NetCDF.

    Retorno:
    -------
    Si mode == 'full': xarray.DataArray con las pendientes globales de la variable.
    Si mode == 'seasonal': dict con 4 xarray.DataArray (invierno, primavera, verano, otoño).
    Si mode == 'monthly': dict con 12 xarray.DataArray, uno para cada mes.
    """
    if mode == 'full':
        # Pendientes totales a través de la dimensión temporal
        slopes = raster_series_regression_2d_slopes(ds, variable_name, step_name=step_name)
        if store:
            save_ndarray_as_netcdf_to_file(ds, slopes, 'slope', f"{store_dir}/{file_name}_linreg_slope_{mode}.nc")
        return slopes

    elif mode == 'seasonal':
        # Diccionario para almacenar las pendientes estacionales
        seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11]
        }
        seasonal_slopes = {}
        for season, months in seasons.items():
            season_ds = ds.sel(time=ds['time.month'].isin(months))
            seasonal_slopes[season] = raster_series_regression_2d_slopes(season_ds, variable_name, step_name=f"{step_name} - {season}")
            if store:
                save_ndarray_as_netcdf_to_file(ds, seasonal_slopes[season], 'slope', f"{store_dir}/{file_name}_linreg_slope_{mode}_{season}.nc")
        return seasonal_slopes

    elif mode == 'monthly':
        # Diccionario para almacenar las pendientes mensuales
        monthly_slopes = {}
        for month in range(1, 13):
            month_ds = ds.sel(time=ds['time.month'] == month)
            monthly_slopes[month] = raster_series_regression_2d_slopes(month_ds, variable_name, step_name=f"{step_name} - {month}")
            if store:
                save_ndarray_as_netcdf_to_file(ds, monthly_slopes[month], 'slope', f"{store_dir}/{file_name}_linreg_slope_{mode}_m{month}.nc")
        return monthly_slopes

    else:
        raise ValueError(f"Modo '{mode}' no reconocido. Usa 'full', 'seasonal' o 'monthly'.")

# ocean_tools/visualization/maps.py

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors

def plot_spatial_variable(target, lat, lon, i_width=8, i_height=8, 
                          clim=[0, 40], title="", plot_label='Variable', 
                          cmap='RdBu_r', norm_mode='none'):
    """
    Grafica una sola matriz 2D (lat, lon) en un mapa usando cartopy.
    """
    plt.figure(figsize=(i_width, i_height))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])

    if norm_mode == 'none':
        plt.pcolormesh(lon, lat, target, cmap=cmap)
        plt.clim(clim)
    elif norm_mode == 'log':
        plt.pcolormesh(lon, lat, target, cmap=cmap, norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]))
    elif norm_mode == 'symlog':
        plt.pcolormesh(lon, lat, target, cmap=cmap, 
                       norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1, 
                                              vmin=clim[0], vmax=clim[1]))

    plt.colorbar(label=plot_label)
    plt.title(title, fontsize=10)
    plt.show()


def plot_n_spatial_variable(target_list, lat, lon, nrows, ncols, i_width=8, i_height=8, 
                            clim=[0, 40], titles=[], plot_label='Variable', 
                            cmap='RdBu_r', norm_mode='none'):
    """
    Grafica múltiples matrices 2D (lat, lon) en subplots.
    """
    import numpy as np
    plt.figure(figsize=(i_width * ncols, i_height * nrows))
    for i, target in enumerate(target_list):
        ax = plt.subplot(nrows, ncols, i + 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])

        if norm_mode == 'none':
            plt.pcolormesh(lon, lat, target, cmap=cmap)
            plt.clim(clim)
        elif norm_mode == 'log':
            plt.pcolormesh(lon, lat, target, cmap=cmap, norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]))
        elif norm_mode == 'symlog':
            plt.pcolormesh(lon, lat, target, cmap=cmap, 
                           norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1, 
                                                  vmin=clim[0], vmax=clim[1]))
        plt.colorbar(label=plot_label)

        if len(titles) > i:
            plt.title(titles[i], fontsize=10)

    plt.tight_layout()
    plt.show()


# ocean_tools/visualization/reports.py

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import numpy as np

def plot_n_eof_report(patterns, time_series, max_mins, lat, lon, time_dim, 
                      clim_pattern=[-0.01, 0.01], clim_maxmin=[-5, 5], 
                      cmap='RdBu_r', pat_norm_mode='none', maxmin_norm_mode='none'):
    """
    Genera un reporte con la info de cada EOF:
     - Patrón espacial
     - Serie temporal
     - Campo espacial en el máximo y mínimo de la PC
    """
    n_patterns = len(patterns)
    fig = plt.figure(figsize=(20, n_patterns * 3))
    gs = gridspec.GridSpec(n_patterns, 4, width_ratios=[1, 3, 1, 1])
    
    for i in range(n_patterns):
        # -- EOF pattern
        ax = plt.subplot(gs[i, 0], projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])

        if pat_norm_mode == 'none':
            plt.pcolormesh(lon, lat, patterns[i], cmap=cmap)
            plt.clim(clim_pattern)
        elif pat_norm_mode == 'log':
            plt.pcolormesh(lon, lat, patterns[i], cmap=cmap, 
                           norm=colors.LogNorm(vmin=clim_pattern[0], vmax=clim_pattern[1]))
        elif pat_norm_mode == 'symlog':
            plt.pcolormesh(lon, lat, patterns[i], cmap=cmap,
                           norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1, 
                                                  vmin=clim_pattern[0], vmax=clim_pattern[1]))
        plt.colorbar(label='')
        plt.title(f'EOF {i+1}')

        # -- Time series
        ax = plt.subplot(gs[i, 1])
        plt.plot(time_dim, time_series[i])
        plt.ylabel('Intensity')
        plt.ylim(-5, 5)
        plt.title(f'TS {i+1}')
        min_year = np.datetime_as_string(time_dim[0], unit='Y').astype(int)
        max_year = np.datetime_as_string(time_dim[-1], unit='Y').astype(int)
        for year_line in range(min_year, max_year + 2):
            plt.axvline(x=np.datetime64(f'{year_line}-01-01'), color='r', linestyle='--')

        # -- Max date
        ax = plt.subplot(gs[i, 2], projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])

        if maxmin_norm_mode == 'none':
            plt.pcolormesh(lon, lat, max_mins[i][0], cmap=cmap)
            plt.clim(clim_maxmin)
        elif maxmin_norm_mode == 'log':
            plt.pcolormesh(lon, lat, max_mins[i][0], cmap=cmap, 
                   norm=colors.LogNorm(vmin=clim_maxmin[0], vmax=clim_maxmin[1]))
        elif maxmin_norm_mode == 'symlog':
            plt.pcolormesh(lon, lat, max_mins[i][0], cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1,
                              vmin=clim_maxmin[0], vmax=clim_maxmin[1]))
        plt.colorbar(label='')
        max_date = np.datetime_as_string(time_dim[np.argmax(time_series[i])], unit='M')
        plt.title(f'Max {i+1} - {max_date}')

        # -- Min date
        ax = plt.subplot(gs[i, 3], projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])
        
        if maxmin_norm_mode == 'none':
            plt.pcolormesh(lon, lat, max_mins[i][1], cmap=cmap)
            plt.clim(clim_maxmin)
        elif maxmin_norm_mode == 'log':
            plt.pcolormesh(lon, lat, max_mins[i][1], cmap=cmap, 
                   norm=colors.LogNorm(vmin=clim_maxmin[0], vmax=clim_maxmin[1]))
        elif maxmin_norm_mode == 'symlog':
            plt.pcolormesh(lon, lat, max_mins[i][1], cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1,
                              vmin=clim_maxmin[0], vmax=clim_maxmin[1]))
        plt.colorbar(label='')
        min_date = np.datetime_as_string(time_dim[np.argmin(time_series[i])], unit='M')
        plt.title(f'Min {i+1} - {min_date}')
        
    plt.tight_layout()
    plt.show()