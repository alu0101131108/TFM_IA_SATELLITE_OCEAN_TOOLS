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
