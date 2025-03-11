# ocean_tools/processing/data_prep.py

import numpy as np
from .anomalies import get_deseasonalized_anomaly_ds
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def prepare_dataset_for_analysis(ds, variable_name, use_anomalies=False, anomaly_transform='none'):
    """
    Prepares a dataset for analysis by masking out non-ocean areas and, optionally,
    computing deseasonalized anomalies with additional transformations.

    The function performs the following steps:
      1. Generates an ocean mask by plotting the dataset extent using Cartopy,
         drawing land features, and extracting a binary mask from the figure's RGBA buffer.
      2. Applies the mask to the specified variable, setting values outside the ocean to NaN.
      3. If use_anomalies is True, computes deseasonalized anomalies by subtracting the
         monthly mean from each time step, and then applies a transformation based on anomaly_transform:
            - 'positive': keeps only positive anomaly values.
            - 'negative': keeps only negative anomaly values.
            - 'square': squares the anomaly values.
            - 'abs': takes the absolute value.
            - 'none': no further transformation is applied.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the variable and coordinates (time, lat, lon).
    variable_name : str
        The name of the variable to process (e.g., 'sst' or 'chlor_a').
    use_anomalies : bool, optional
        If True, computes the deseasonalized anomaly for the variable. Default is False.
    anomaly_transform : str, optional
        Transformation to apply on the anomaly:
          'positive', 'negative', 'square', 'abs', or 'none'. Default is 'none'.

    Returns
    -------
    xr.Dataset
        The processed dataset with non-ocean values masked out and, if requested,
        the variable replaced with its deseasonalized (and transformed) anomaly.
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
    Aggregates a variable in a dataset over time according to a specified mode.

    Depending on the mode, the function returns:
      - 'full': A single xarray.DataArray with the aggregation over the entire time dimension.
      - 'seasonal': A dictionary with keys 'winter', 'spring', 'summer', and 'autumn', each containing an xarray.DataArray aggregated over the respective months.
      - 'monthly': A dictionary with keys 1 to 12 (months) and corresponding xarray.DataArrays.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the variable and a time coordinate.
    variable_name : str
        The name of the variable to aggregate.
    mode : str, optional
        Aggregation mode: 'full', 'seasonal', or 'monthly'. Default is 'full'.
    agg_type : str, optional
        Type of aggregation (e.g., 'mean', 'std') supported by xarray methods. Default is 'mean'.

    Returns
    -------
    xr.DataArray or dict
        If mode is 'full', returns an xarray.DataArray with the aggregated variable.
        If mode is 'seasonal' or 'monthly', returns a dictionary mapping the period (season or month)
        to the corresponding xarray.DataArray.
    
    Raises
    ------
    ValueError
        If the specified aggregation type is not supported or if the mode is unrecognized.
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
