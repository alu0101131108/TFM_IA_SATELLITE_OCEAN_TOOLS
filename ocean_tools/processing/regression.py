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
