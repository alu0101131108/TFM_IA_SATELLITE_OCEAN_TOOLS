# ocean_tools/processing/anomalies.py

import xarray as xr

def get_deseasonalized_anomaly_ds(ds, variable_name):
    """
    Computes the deseasonalized anomaly for a specified variable in an xarray Dataset.

    For each month (from 1 to 12), the function calculates the mean of the variable over all 
    time steps corresponding to that month. Then, for every time step in the dataset, it subtracts 
    the mean value of the corresponding month, effectively removing the seasonal cycle.

    Parameters
    ----------
    * ds : xr.Dataset
        * Input dataset with at least 'time' and spatial dimensions (e.g., 'lat' and 'lon').
    * Variable_name : str
        * Name of the variable (e.g., 'sst') for which the deseasonalized anomaly is to be computed.

    Returns
    -------
    * xr.Dataset
        * The input dataset with the specified variable updated to represent its deseasonalized anomaly.
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
