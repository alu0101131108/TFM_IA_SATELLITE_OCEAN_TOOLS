# ocean_tools/io/writers.py
import xarray as xr
import os
import pickle

def save_ndarray_as_netcdf_to_file(ref_ds, ndarray, var_name, file_path):
    """
    Save a NumPy ndarray as a NetCDF file using a reference xarray Dataset for coordinate information.

    This function creates an xarray DataArray from the provided ndarray and uses the 'time',
    'lat', and 'lon' coordinates from the reference dataset (ref_ds) to construct the DataArray.
    The ndarray must be either 3-dimensional (with dimensions [time, lat, lon]) or 2-dimensional 
    (with dimensions [lat, lon]). If the specified file_path already exists, an incremental suffix
    is appended to the filename.

    Parameters
    ----------
    * ref_ds : xr.Dataset
        * A reference dataset containing the coordinate variables 'time', 'lat', and 'lon'.
    * ndarray : np.ndarray
        * The NumPy array to be saved. Must have shape (time, lat, lon) or (lat, lon).
    * var_name : str
        * The name to assign to the variable in the NetCDF file.
    * file_path : str
        * The desired output file path for the NetCDF file.

    Returns
    -------
    None
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
    """
    Save a Python variable as a pickle file in the specified directory.

    This function constructs a file path from the provided store_dir and file_name (appending a '.pkl'
    extension). If a file with that name already exists, it appends an incremental suffix to generate a 
    unique filename, then saves the variable using pickle.

    Parameters
    ----------
    * variable : Any
        * The Python variable (pickle-able object) to be stored.
    * store_dir : str
        * The directory where the pickle file will be saved.
    * file_name : str
        * The desired base name for the pickle file (without the .pkl extension).

    Returns
    -------
    None
    """
    file_path = os.path.join(store_dir, f'{file_name}.pkl')
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base}_{counter}{ext}"
        counter += 1
    with open(new_file_path, 'wb') as f:
        pickle.dump(variable, f)