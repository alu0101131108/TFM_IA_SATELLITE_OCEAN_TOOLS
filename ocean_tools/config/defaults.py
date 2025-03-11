"""
Module: ocean_tools/config/defaults.py

This module defines default configuration parameters for ocean_tools, including geographic 
regions, variable descriptions, climatology defaults, and a set of lambda functions for calculating 
cluster features from segmented cluster data. These features characterize the spatial and temporal 
extent and shape of clusters derived from satellite data.

It includes the following constants:

* REGIONS: A dictionary defining geographic regions with their latitude and longitude boundaries.
* VARIABLES: A mapping of variable codes to their descriptive names.
* CLIM_DEFAULTS: Default climatology limits for each variable.
* CLUSTER_FEATURES_LAMBDA: Dictionary of lambda functions for calculating various cluster features.
    * 'time_length': Difference between the maximum and minimum time index + 1.
    * 'lat_length': Difference between the maximum and minimum latitude index + 1.
    * 'lon_length': Difference between the maximum and minimum longitude index + 1.
    * 'cluster_size': Total number of grid points in the cluster.
    * 'cluster_volume': Volume of the bounding box covering the cluster.
    * 'cluster_compactness': Ratio of the actual cluster size to the volume of its bounding box.
    * 'cluster_eccentricity': Relative difference between the maximum and minimum spans across dimensions.
    * 'cluster_dispersion': Average Euclidean distance from the cluster centroid.
"""

import numpy as np
import xarray as xr

# Dictionary defining geographic regions with their latitude and longitude boundaries.
REGIONS = {
    'cclme': {
        'lat': [5, 40],
        'lon': [-30, -5]
    },
    'weak_permanent_uw': {
        'lat': [26, 35],
        'lon': [-30, -5]
    },
    'permanent_uw': {
        'lat': [21, 26],
        'lon': [-30, -12]
    },
    'mauritania_senegalese_uw': {
        'lat': [12, 19],
        'lon': [-30, -15]
    }
}

# Mapping of variable codes to their descriptive names.
VARIABLES = {
    'sst': 'Sea Surface Temperature',
    'chlor_a': 'Chlorophyll-a'
}

# Default climatology limits for each variable.
CLIM_DEFAULTS = {
    'sst': [0, 35],
    'chlor_a': [0.01, 20]
}

# Calculate the average Euclidean distance of all cluster points from the cluster's centroid.
def cluster_dispersion_func(arr):
    if not np.any(~np.isnan(arr)):
        return np.nan
    # Get indices for each dimension where the cluster is present.
    indices = np.array(np.where(~np.isnan(arr)))  # Shape: (ndim, n_points)
    # Compute the centroid along each dimension.
    centroid = np.mean(indices, axis=1)  # Shape: (ndim,)
    # Compute the Euclidean distance from the centroid for each point.
    distances = np.sqrt(np.sum((indices - centroid[:, None])**2, axis=0))
    return np.mean(distances)

# Dictionary of lambda functions for calculating various cluster features.
# Each lambda function expects a 3D numpy array (from a segmented cluster) where non-NaN values indicate cluster points.
CLUSTER_FEATURES_LAMBDA = {
    # 'time_length': Difference between the maximum and minimum time index + 1.
    'time_length': lambda arr: (np.ptp(np.where(~np.isnan(arr))[0]) + 1)
        if np.any(~np.isnan(arr)) else np.nan,
    
    # 'lat_length': Difference between the maximum and minimum latitude index + 1.
    'lat_length': lambda arr: (np.ptp(np.where(~np.isnan(arr))[1]) + 1)
        if np.any(~np.isnan(arr)) else np.nan,
    
    # 'lon_length': Difference between the maximum and minimum longitude index + 1.
    'lon_length': lambda arr: (np.ptp(np.where(~np.isnan(arr))[2]) + 1)
        if np.any(~np.isnan(arr)) else np.nan,
    
    # 'cluster_size': Total number of grid points in the cluster.
    'cluster_size': lambda arr: np.sum(~np.isnan(arr)),
    
    # 'cluster_volume': Volume of the bounding box covering the cluster.
    'cluster_volume': lambda arr: (
        (np.ptp(np.where(~np.isnan(arr))[0]) + 1) *
        (np.ptp(np.where(~np.isnan(arr))[1]) + 1) *
        (np.ptp(np.where(~np.isnan(arr))[2]) + 1)
    ) if np.any(~np.isnan(arr)) else np.nan,
    
    # 'cluster_compactness': Ratio of the actual cluster size to the volume of its bounding box.
    'cluster_compactness': lambda arr: (
        np.sum(~np.isnan(arr)) /
        ((np.ptp(np.where(~np.isnan(arr))[0]) + 1) *
         (np.ptp(np.where(~np.isnan(arr))[1]) + 1) *
         (np.ptp(np.where(~np.isnan(arr))[2]) + 1))
    ) if np.any(~np.isnan(arr)) else np.nan,
    
    # 'cluster_eccentricity': Relative difference between the maximum and minimum spans across dimensions.
    'cluster_eccentricity': lambda arr: (
        (max(np.ptp(np.where(~np.isnan(arr))[0]) + 1,
             np.ptp(np.where(~np.isnan(arr))[1]) + 1,
             np.ptp(np.where(~np.isnan(arr))[2]) + 1) -
         min(np.ptp(np.where(~np.isnan(arr))[0]) + 1,
             np.ptp(np.where(~np.isnan(arr))[1]) + 1,
             np.ptp(np.where(~np.isnan(arr))[2]) + 1)
        ) / max(np.ptp(np.where(~np.isnan(arr))[0]) + 1,
               np.ptp(np.where(~np.isnan(arr))[1]) + 1,
               np.ptp(np.where(~np.isnan(arr))[2]) + 1)
    ) if np.any(~np.isnan(arr)) else np.nan,
    
    # 'cluster_dispersion': Average Euclidean distance from the cluster centroid.
    'cluster_dispersion': lambda arr: cluster_dispersion_func(arr)
}
