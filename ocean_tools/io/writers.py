# ocean_tools/io/writers.py
import xarray as xr
import os
import pickle

def save_ndarray_as_netcdf_to_file(ref_ds, ndarray, var_name, file_path):
    """
    Save an ndarray as a NetCDF file at the specified path.

    Parameters:
    -----------
    ref_ds : xarray.Dataset
        Reference dataset containing latitude, longitude, and time coordinates.
    ndarray : np.ndarray
        The data array to save.
    var_name : str
        The name of the variable to save in the NetCDF file.
    file_path : str
        The output NetCDF file path.
    """
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path

    while os.path.exists(new_file_path):
        new_file_path = f"{base}_{counter}{ext}"
        counter += 1

    if len(ndarray.shape) == 3:
        ds_xarray = xr.DataArray(ndarray, dims=['time', 'lat', 'lon'], coords={'time': ref_ds.time, 'lat': ref_ds.lat, 'lon': ref_ds.lon}, name=var_name)
    elif len(ndarray.shape) == 2:
        ds_xarray = xr.DataArray(ndarray, dims=['lat', 'lon'], coords={'lat': ref_ds.lat, 'lon': ref_ds.lon}, name=var_name)
    else:
        raise ValueError("The dataset must have dimensions (lat, lon) or (time, lat, lon).")
    ds_xarray.to_netcdf(new_file_path)


def store_pickle_variable(variable, store_dir, file_name):
    file_path = os.path.join(store_dir, f'{file_name}.pkl')
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base}_{counter}{ext}"
        counter += 1
    with open(new_file_path, 'wb') as f:
        pickle.dump(variable, f)