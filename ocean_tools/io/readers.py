# ocean_tools/io/readers.py

import xarray as xr

def get_xarray_from_file(file_path):
    """
    Loads a NetCDF file and returns the resulting xarray Dataset.

    Parameters
    ----------
    file_path : str
        The full file path to the NetCDF file.

    Returns
    -------
    xr.Dataset
        The xarray Dataset loaded from the specified file.
    """
    ds = xr.open_dataset(file_path)
    return ds
