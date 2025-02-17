# ocean_tools/io/readers.py

import xarray as xr

def get_xarray_from_file(file_path):
    """
    Carga un archivo NetCDF y devuelve el Dataset de xarray resultante.
    """
    ds = xr.open_dataset(file_path)
    return ds
