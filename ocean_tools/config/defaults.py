# ocean_tools/config/defaults.py

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

VARIABLES = {
    'sst': 'Sea Surface Temperature',
    'chlor_a': 'Chlorophyll-a'
}

CLIM_DEFAULTS = {
    'sst': [0, 35],
    'chlor_a': [0.01, 20]
}

import numpy as np
CLUSTER_FEATURES_LAMBDA = {
    # Time length: difference between maximum and minimum time index + 1.
    'time_length': lambda arr: (np.ptp(np.where(~np.isnan(arr))[0]) + 1)
        if np.any(~np.isnan(arr)) else np.nan,
    
    # Latitude span.
    'lat_length': lambda arr: (np.ptp(np.where(~np.isnan(arr))[1]) + 1)
        if np.any(~np.isnan(arr)) else np.nan,
    
    # Longitude span.
    'lon_length': lambda arr: (np.ptp(np.where(~np.isnan(arr))[2]) + 1)
        if np.any(~np.isnan(arr)) else np.nan,
    
    # Cluster size: simply the number of non-NaN grid points.
    'cluster_size': lambda arr: np.sum(~np.isnan(arr)),
    
    # Cluster volume: the volume of the bounding box covering the cluster.
    'cluster_volume': lambda arr: (
        (np.ptp(np.where(~np.isnan(arr))[0]) + 1) *
        (np.ptp(np.where(~np.isnan(arr))[1]) + 1) *
        (np.ptp(np.where(~np.isnan(arr))[2]) + 1)
    ) if np.any(~np.isnan(arr)) else np.nan,
    
    # Cluster compactness: fraction of the bounding box that is actually filled.
    'cluster_compactness': lambda arr: (
        np.sum(~np.isnan(arr)) /
        ((np.ptp(np.where(~np.isnan(arr))[0]) + 1) *
         (np.ptp(np.where(~np.isnan(arr))[1]) + 1) *
         (np.ptp(np.where(~np.isnan(arr))[2]) + 1))
    ) if np.any(~np.isnan(arr)) else np.nan,
    
    # Cluster eccentricity: (max(axis lengths) - min(axis lengths)) / max(axis lengths)
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
    
    # Cluster density: here defined as cluster_size divided by cluster_volume.
    # (In this binary case it is identical to compactness, but could be adapted.)
    'cluster_density': lambda arr: (
        np.sum(~np.isnan(arr)) /
        ((np.ptp(np.where(~np.isnan(arr))[0]) + 1) *
         (np.ptp(np.where(~np.isnan(arr))[1]) + 1) *
         (np.ptp(np.where(~np.isnan(arr))[2]) + 1))
    ) if np.any(~np.isnan(arr)) else np.nan,
}