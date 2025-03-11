# ocean_tools/processing/merges.py

import os
import xarray as xr

def merge_variable_days_preprocess(dataset_dir, file, lat, lon, variable_name):
    """
    Loads a NetCDF file, expands its 'time' dimension, filters the dataset to a specified region, 
    and retains only the specified variable.

    The file name is expected to contain a date in the format where the date appears after the first dot
    and consists of eight digits (YYYYMMDD). This date is used to set the 'time' coordinate of the dataset.

    Parameters
    ----------
    dataset_dir : str
        The directory where the NetCDF file is located.
    file : str
        The name of the NetCDF file.
    lat : list or tuple of float
        A list or tuple containing the minimum and maximum latitudes [min_lat, max_lat] for filtering.
    lon : list or tuple of float
        A list or tuple containing the minimum and maximum longitudes [min_lon, max_lon] for filtering.
    variable_name : str
        The name of the variable to retain in the dataset.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing only the specified variable, with the 'time' dimension expanded 
        (and set to a single date extracted from the file name) and filtered to the specified geographic region.
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
