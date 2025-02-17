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
