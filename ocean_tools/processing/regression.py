# ocean_tools/processing/regression.py

import numpy as np
from sklearn import linear_model

def raster_series_regression_2d_slopes(ds, variable_name, step_name="Step"):
    """
    Computes linear regression slopes at each (lat, lon) grid point from the time series of the specified variable.

    For each spatial point in the dataset, the function fits a linear regression model using the time coordinate
    (converted to an integer type) as the independent variable and the variable values as the dependent variable.
    It returns a 2D numpy array of slopes with the same spatial dimensions as the dataset.

    Parameters
    ----------
    * ds : xr.Dataset
        * Input dataset that contains the variable as well as 'time', 'lat', and 'lon' coordinates.
    * variable_name : str
        * The name of the variable on which to perform the regression (e.g., 'sst').
    * step_name : str, optional
        * A label used to display progress during computation. Default is "Step".

    Returns
    -------
    * np.ndarray
        * 2D numpy array of slopes with shape (len(lat), len(lon)).
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
    Calculates regression slopes for a dataset based on the specified temporal mode.

    * Depending on the mode, this function computes:
        * 'full': A single 2D spatial array of slopes computed over the entire time series.
        * 'seasonal': A dictionary with keys ('winter', 'spring', 'summer', 'autumn'), each containing a 2D
                    spatial array of slopes computed over the corresponding season.
        * 'monthly': A dictionary with keys 1 to 12, each containing a 2D spatial array of slopes computed
                   for the corresponding month.

    Optionally, the computed slopes are saved to NetCDF files.

    Parameters
    ----------
    * ds : xr.Dataset
        * Input dataset containing the variable and coordinates (time, lat, lon).
    * variable_name : str
        * The name of the variable to process (e.g., 'sst').
    * mode : str, optional
        * Temporal mode of computation: 'full', 'seasonal', or 'monthly'. Default is 'full'.
    * step_name : str, optional
        * Label used for progress display. Default is "Step".
    * store : bool, optional
        * If True, the computed slopes are saved as NetCDF files. Default is False.
    * store_dir : str, optional
        * Directory where the NetCDF files will be stored if store is True.
    * file_name : str, optional
        * Base file name for the stored NetCDF files if store is True.

    Returns
    -------
    * xr.DataArray or dict
        * If mode is 'full', returns an xarray.DataArray with slopes.
        * If mode is 'seasonal' or 'monthly', returns a dictionary mapping season/month to xarray.DataArray of slopes.
    * dict
        * A dictionary of additional information (e.g., number of clusters may be added in future versions).
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
