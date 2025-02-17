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
