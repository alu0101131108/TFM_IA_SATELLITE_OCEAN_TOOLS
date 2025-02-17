from ocean_tools.io.readers import get_xarray_from_file
from ocean_tools.config.defaults import REGIONS
from ocean_tools.processing.data_prep import prepare_dataset_for_analysis, time_aggregator
from ocean_tools.visualization.maps import plot_spatial_variable, plot_n_spatial_variable

REGION = REGIONS['cclme']
DATASETS = {
    'sst': "./data/exports/AQUA_MODIS_MONTHLY.2002-08-01_2024-11-01.nc",
    'chlor_a': "./data/exports/AQUA_MODIS_MONTHLY_CHLOR.2002-08-01_2024-11-01.nc"
}

ds_sst_raw = get_xarray_from_file(DATASETS['sst']).sel(
    time=slice('2002-08-01', '2024-07-01'), 
    lat=slice(REGION['lat'][1], REGION ['lat'][0]),
    lon=slice(REGION['lon'][0], REGION ['lon'][1])
)

ds_sst = prepare_dataset_for_analysis(ds_sst_raw.copy(), 'sst', use_anomalies=False)

ds_sst_month = ds_sst.sel(time='2002-08-01')['sst']
plot_n_spatial_variable(
    [ds_sst_month],
    ds_sst.lat,
    ds_sst.lon,
    nrows = 1,
    ncols = 1,
    i_width = 8,
    i_height = 8,
    clim=[15, 35], 
    titles = ['SST 2002-08'],
    plot_label = 'Sea Surface Temperature (°C)',
    cmap = 'RdBu_r',
    norm_mode = 'none'
)