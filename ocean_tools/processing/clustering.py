"""
Module: ocean_tools/processing/clustering.py

This module provides functions for processing anomalies and clustering on spatiotemporal 
oceanographic datasets. It includes methods to segment anomalies, combine segmentation from 
multiple variables, perform connected component labeling (using both flood-fill and DBSCAN-like 
approaches), extract cluster features, aggregate experiment-level features, and compute 
clustering quality metrics.
"""

import xarray as xr
import numpy as np

# Anomaly segmentation into components.
def segment_anomalies(ds_anom, variable_name, anomaly_threshold):
    """
    Generates a segmentation mask for a single dataset by thresholding the absolute anomaly values.

    For each grid point, if the absolute value of the specified variable is greater than or equal 
    to the anomaly_threshold and is not null, it is labeled as 1; otherwise, it is set to np.nan.

    Parameters
    ----------
    * ds_anom : xr.Dataset
        * Input dataset containing the variable to be segmented (with dimensions like time, lat, lon).
    * variable_name : str
        * Name of the variable to use for segmentation (e.g., 'sst' or 'chlor_a').
    * anomaly_threshold : float
        * Threshold value for determining a relevant anomaly.

    Returns
    -------
    * xr.DataArray
        * A segmentation mask (DataArray) with the same dimensions as ds_anom[variable_name],
        where relevant grid points are 1 and non-relevant ones are np.nan.
    """
    ds_anom_var = abs(ds_anom[variable_name])
    ds_segmented = xr.where((ds_anom_var >= anomaly_threshold) & ds_anom_var.notnull(), 1, np.nan)
    return ds_segmented

def segment_n_variable_anomalies(ds_anom_list, var_name_list, anomaly_threshold_list, union_type='and'):
    """
    Computes a combined segmentation mask for multiple datasets/variables based on specified thresholds.

    For each dataset in ds_anom_list, a binary mask is created where grid points meeting or 
    exceeding the corresponding anomaly threshold are marked as 1 (and others as np.nan). These 
    masks are then combined element-wise using either an "and" (intersection) or "or" (union) logic. 
    The final segmentation mask has the same shape, coordinates, and dimensions as the first dataset's variable.

    Parameters
    ----------
    * ds_anom_list : list of xr.Dataset or xr.DataArray
        * List of datasets (or DataArrays) containing the variables to process. Each is assumed to have dimensions (time, lat, lon).
    * var_name_list : list of str
        * List of variable names corresponding to each dataset in ds_anom_list.
    * anomaly_threshold_list : list of float
        * List of anomaly thresholds for each dataset.
    * union_type : str, optional
        * Logic for combining masks: 'and' to require all datasets to exceed thresholds, 'or' to allow any.
        Default is 'and'.

    Returns
    -------
    * xr.DataArray
        * A combined segmentation mask with values 1 where the union criteria are met, np.nan otherwise.
    """
    # Process the first dataset directly using its numpy array.
    first_arr = np.abs(ds_anom_list[0][var_name_list[0]].values)
    first_thresh = anomaly_threshold_list[0]
    combined_bool = (first_arr >= first_thresh) & np.isfinite(first_arr)
    
    # Process each subsequent dataset.
    for i in range(1, len(ds_anom_list)):
        arr = np.abs(ds_anom_list[i][var_name_list[i]].values)
        thresh = anomaly_threshold_list[i]
        current_bool = (arr >= thresh) & np.isfinite(arr)
        
        if union_type == 'and':
            combined_bool = combined_bool & current_bool
        elif union_type == 'or':
            combined_bool = combined_bool | current_bool
        else:
            raise ValueError('Invalid union type: choose "and" or "or"')
    

    combined_seg = np.where(combined_bool, 1, np.nan)
    combined_mask = xr.DataArray(
        combined_seg,
        coords=ds_anom_list[0][var_name_list[0]].coords,
        dims=ds_anom_list[0][var_name_list[0]].dims
    )
    return combined_mask

def label_connected_components(ds_segmented, eps_time=1, eps_lat=1, eps_lon=1, min_cluster_size=5):
    """
    Labels connected components in a segmentation mask using a flood-fill approach.

    The function performs a flood-fill (depth-first search) over the 3D segmentation mask 
    (dimensions: time, lat, lon). It assigns a unique positive integer label to each connected 
    component (cluster) that meets the min_cluster_size requirement; clusters smaller than this 
    threshold are marked as noise (label 0).

    Parameters
    ----------
    * ds_segmented : xr.DataArray
        * A segmentation mask (with values 1 for relevant anomalies and np.nan for others) with dimensions (time, lat, lon).
    * eps_time : int, optional
        * Maximum index difference in the time dimension for neighbor connectivity. Default is 1.
    * eps_lat : int, optional
        * Maximum index difference in the latitude dimension for neighbor connectivity. Default is 1.
    * eps_lon : int, optional
        * Maximum index difference in the longitude dimension for neighbor connectivity. Default is 1.
    * min_cluster_size : int, optional
        * Minimum number of points required for a cluster to be retained. Clusters smaller than this are labeled as noise (0). Default is 5.

    Returns
    -------
    * ds_labels : xr.DataArray
        * A DataArray of the same shape and coordinates as ds_segmented, with each cluster assigned a unique positive integer label and noise points labeled as 0.
    * n_clusters : int
        * The number of clusters that meet the min_cluster_size requirement.
    """
    seg_arr = ds_segmented.values
    T, LAT, LON = seg_arr.shape
    
    # Initialize label array and mark NaN positions.
    labels = np.full(seg_arr.shape, np.nan, dtype=float)
    nan_mask = np.isnan(seg_arr)
    labels[nan_mask] = np.nan
    
    # Track visited points; mark NaN as visited.
    visited = np.zeros(seg_arr.shape, dtype=bool)
    visited[nan_mask] = True

    current_label = 1
    
    def in_bounds(t, i, j):
        """Check if indices (t, i, j) are within the array bounds."""
        return (0 <= t < T) and (0 <= i < LAT) and (0 <= j < LON)

    # Iterate over all points in the 3D array.
    for t in range(T):
        for i in range(LAT):
            for j in range(LON):
                if visited[t, i, j]: continue # Skip visited points.
                
                # Start flood-fill from unvisited point.
                stack = [(t, i, j)]
                component_points = []
                while stack:
                    tt, ii, jj = stack.pop()
                    if not in_bounds(tt, ii, jj) or visited[tt, ii, jj]: continue
                    visited[tt, ii, jj] = True
                    component_points.append((tt, ii, jj))

                    # Check neighbors within the specified eps windows.
                    for dt in range(-eps_time, eps_time + 1):
                        for di in range(-eps_lat, eps_lat + 1):
                            for dj in range(-eps_lon, eps_lon + 1):
                                nt = tt + dt
                                ni = ii + di
                                nj = jj + dj
                                if not in_bounds(nt, ni, nj) or visited[nt, ni, nj]: continue # Skip out-of-bounds or visited points.
                                
                                # Assign labels based on cluster size.
                                stack.append((nt, ni, nj))
                
                # Assign labels based on cluster size.
                cluster_size = len(component_points)
                if cluster_size < min_cluster_size:
                    # If too small, assign outlier label (-1) to all points in this group.
                    for (tt, ii, jj) in component_points:
                        labels[tt, ii, jj] = 0
                else:
                    # Otherwise, assign a unique label to the entire cluster.
                    for (tt, ii, jj) in component_points:
                        labels[tt, ii, jj] = current_label
                    current_label += 1
                    print(f"Identified cluster #{current_label-1} ({cluster_size})")
                    print(f"Visited: {(np.sum(visited) / visited.size) * 100:.2f}%")
    
    # Wrap the resulting label array into an xarray DataArray with the same coords/dims.
    ds_labels = xr.DataArray(labels, coords=ds_segmented.coords, dims=ds_segmented.dims)
    n_labels = current_label - 1
    return ds_labels, n_labels

# STDBSCAN like clustering for connected component labeling.
def label_connected_components_dbscan(
    ds_segmented, 
    eps_time=1, 
    eps_lat=1, 
    eps_lon=1, 
    min_neighbors=5, 
    min_cluster_size=0
):
    """
    Performs DBSCAN-like clustering on a segmentation mask using a flood-fill approach.

    Each point in the segmentation mask (with value 1 for relevant anomalies) is considered.
    A point is considered a core point if its spatiotemporal neighborhood (defined by eps_time, eps_lat, and eps_lon)
    contains at least min_neighbors points. The algorithm then expands the cluster via these core points.
    After cluster expansion, if the total number of points in the cluster is below min_cluster_size, the cluster is discarded (labeled as noise, -1).

    Parameters
    ----------
    * ds_segmented : xr.DataArray
        * Segmentation mask with dimensions (time, lat, lon) where relevant points have value 1 and non-relevant or missing values are 0 or np.nan.
    * eps_time : int, optional
        * Maximum index difference in the time dimension for neighbor search. Default is 1.
    * eps_lat : int, optional
        * Maximum index difference in the latitude dimension for neighbor search. Default is 1.
    * eps_lon : int, optional
        * Maximum index difference in the longitude dimension for neighbor search. Default is 1.
    * min_neighbors : int, optional
        * Minimum number of points (including the point itself) required for a point to be a core point. Default is 5.
    * min_cluster_size : int, optional
        * Minimum number of points required for a cluster to be retained. Clusters smaller than this will be labeled as noise (-1). Default is 0.

    Returns
    -------
    * ds_labels : xr.DataArray
        * A DataArray with the same dimensions as ds_segmented, where each valid cluster is assigned a unique positive integer label and noise points are labeled as -1.
    * n_clusters : int
        * The number of clusters that meet the min_cluster_size requirement.
    * n_discarded : int
        * The number of clusters discarded due to being smaller than min_cluster_size.
    """
    # Get the underlying NumPy array.
    seg_arr = ds_segmented.values
    T, LAT, LON = seg_arr.shape
    
    # Initialize the labels array: 0 means "unassigned".
    labels = np.full(seg_arr.shape, 0, dtype=float)
    # Propagate NaNs.
    nan_mask = np.isnan(seg_arr)
    labels[nan_mask] = np.nan
    
    # Boolean array to track visited points; mark NaNs as visited.
    visited = np.zeros(seg_arr.shape, dtype=bool)
    visited[nan_mask] = True
    
    current_label = 0
    
    def in_bounds(t, i, j):
        return (0 <= t < T) and (0 <= i < LAT) and (0 <= j < LON)
    
    def region_query(t, i, j):
        """
        Returns a list of (t, i, j) tuples for all relevant points within the spatiotemporal
        neighborhood of (t, i, j) defined by eps_time, eps_lat, and eps_lon.
        """
        neighbors = []
        for dt in range(-eps_time, eps_time + 1):
            for di in range(-eps_lat, eps_lat + 1):
                for dj in range(-eps_lon, eps_lon + 1):
                    nt = t + dt
                    ni = i + di
                    nj = j + dj
                    if in_bounds(nt, ni, nj):
                        if seg_arr[nt, ni, nj] == 1:
                            neighbors.append((nt, ni, nj))
        return neighbors
    
    # Iterate over every point in the spatiotemporal grid.
    for t in range(T):
        for i in range(LAT):
            for j in range(LON):
                if visited[t, i, j]:
                    continue
                if seg_arr[t, i, j] != 1:
                    visited[t, i, j] = True
                    continue
                
                # Mark the starting point as visited.
                visited[t, i, j] = True
                neighbors = region_query(t, i, j)
                
                # If this point is not a core point, mark it as noise.
                if len(neighbors) < min_neighbors:
                    labels[t, i, j] = -1
                else:
                    # Start a new cluster.
                    current_label += 1
                    labels[t, i, j] = current_label
                    seed_set = list(neighbors)
                    # List to collect all indices belonging to this cluster.
                    cluster_points = [(t, i, j)]
                    
                    # Expand the cluster.
                    while seed_set:
                        (qt, qi, qj) = seed_set.pop(0)
                        if not visited[qt, qi, qj]:
                            visited[qt, qi, qj] = True
                            q_neighbors = region_query(qt, qi, qj)
                            if len(q_neighbors) >= min_neighbors:
                                # If the point is a core point, add its neighbors.
                                for qn in q_neighbors:
                                    if qn not in seed_set:
                                        seed_set.append(qn)
                        # Assign the cluster label if not already assigned.
                        if labels[qt, qi, qj] == 0:
                            labels[qt, qi, qj] = current_label
                        cluster_points.append((qt, qi, qj))
                    
                    # After expansion, check the cluster size.
                    if len(cluster_points) < min_cluster_size:
                        # Discard the cluster: mark all its points as noise (-1).
                        for (ct, ci, cj) in cluster_points:
                            labels[ct, ci, cj] = -1
                        print(f"Discarded cluster #{current_label} ({len(cluster_points)})")
                    else:
                        print(f"Identified cluster #{current_label} ({len(cluster_points)})")
                        print(f"Visited: {((np.sum(visited)-nan_mask.sum())/(visited.size-nan_mask.sum())) * 100:.2f}%")

    # Wrap the label array in an xarray DataArray.
    ds_labels = xr.DataArray(labels, coords=ds_segmented.coords, dims=ds_segmented.dims)
    # Count clusters that remain (labels > 0).
    unique_labels = np.unique(labels[~np.isnan(labels)])
    valid_clusters = [lab for lab in unique_labels if lab > 0]
    n_clusters = len(valid_clusters)
    n_discarded = current_label - n_clusters
    
    return ds_labels, n_clusters, n_discarded


import time

def calculate_features_from_extractors (clusters, cluster_extractor, experiment_name = None):
    """
    Calculates cluster features for each unique cluster in a labeled clustering result.

    The function iterates over unique cluster labels (ignoring non-positive labels and NaNs),
    extracts each cluster as a binary mask, and applies each transformation from the cluster_extractor 
    dictionary to compute cluster features. The resulting features for each cluster are stored in a dictionary.

    Parameters
    ----------
    * clusters : xr.DataArray
        * A labeled clustering result with dimensions (time, lat, lon). Valid clusters have positive integer labels.
    * cluster_extractor : dict
        * A dictionary of lambda functions mapping feature names to functions that take a numpy array 
        (extracted from a cluster) and return a feature value.
    * experiment_name : str, optional
        * An optional name for the experiment, used for logging purposes.

    Returns
    -------
    * dict
        * A dictionary mapping cluster IDs (as int) to dictionaries of computed features.
    """
    time_start = time.time()

    # Get the unique cluster labels from the clusters array.
    # Ignore NaN and non-positive labels.
    unique_labels = np.unique(clusters.values)
    unique_labels = unique_labels[~np.isnan(unique_labels)]
    unique_labels = unique_labels[unique_labels > 0]

    featurized_clusters = {}

    # Iterate over each cluster of the experiment.
    for cluster_label in unique_labels:
        # Extract the cluster; this will have the original label for the cluster.
        cluster = clusters.copy().where(clusters == cluster_label)
        # For evaluation, treat all non-NaN values as belonging to one cluster (binary mask).
        cluster_values = cluster.values
        
        cluster_features = {}
        # Calculate features for given cluster.
        for feature_name, transformation in cluster_extractor.items():
            cluster_features[feature_name] = transformation(cluster_values)
        
        featurized_clusters[int(cluster_label)] = cluster_features

    # print completed percentage and accumulated run time.
    print(f"Completed {experiment_name} | Run time: {int((time.time()-time_start) / 60)} minutes.")

    return featurized_clusters

import numpy as np

def extract_experiment_aggregated_features(exp_feature):
    """
    Aggregates cluster-level features from an experiment into experiment-level statistics.

    For each feature present in the clusters, this function computes the mean, standard deviation,
    minimum, and maximum across all clusters, and returns these as a dictionary along with the experiment
    name, file name, and total number of clusters.

    Parameters
    ----------
    * exp_feature : dict
        * Dictionary with keys:
            * 'experiment_name': Name of the experiment.
            * 'file_name': File name associated with the experiment.
            * 'features': Dictionary mapping cluster IDs to dictionaries of cluster features.

    Returns
    -------
    * dict
        * A dictionary containing aggregated experiment-level features.
    """
    cluster_features = exp_feature['features']
    cluster_ids = list(cluster_features.keys())
    n_clusters = len(cluster_ids)
    
    # Initialize a dictionary to hold aggregated metrics.
    metrics = {
        'experiment_name': exp_feature['experiment_name'],
        'file_name': exp_feature['file_name'],
        'num_clusters': n_clusters
    }
    
    # Get the list of feature keys from the first cluster (assumed uniform).
    sample_cluster = next(iter(cluster_features.values()))
    feature_keys = sample_cluster.keys()
    
    # For each feature, gather the value from every cluster.
    for key in feature_keys:
        values = np.array([cluster_features[c][key] for c in cluster_ids])
        metrics[f'{key}_mean'] = float(np.mean(values))
        metrics[f'{key}_std']  = float(np.std(values))
        metrics[f'{key}_min']  = float(np.min(values))
        metrics[f'{key}_max']  = float(np.max(values))
    
    return metrics


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

def compute_clustering_metrics(clusters_xr):
    """
    Computes quality metrics for a clustering result stored in an xarray DataArray.

    The function extracts valid (non-NaN and non-noise) grid points from the clustering result,
    constructs a feature matrix from their indices (time, lat, lon), and computes the silhouette score,
    Davies-Bouldin index, and Calinski-Harabasz index. It also calculates the noise ratio as the 
    proportion of points labeled as noise (-1).

    Parameters
    ----------
    * clusters_xr : xr.DataArray
        * Clustering output with dimensions (time, lat, lon). Valid points have positive integer labels,
        noise points are labeled -1, and missing data are NaN.

    Returns
    -------
    * dict
        * A dictionary containing the following keys:
            *  'silhouette': Silhouette score (higher is better).
            *  'davies_bouldin': Davies-Bouldin index (lower is better)
            *  'calinski_harabasz': Calinski-Harabasz index (higher is better).
            *  'noise_ratio': Proportion of unclustered points (lower is better).
        * If metrics are not computable (e.g., fewer than 2 clusters), their values will be np.nan.
    """
    # Extract the underlying numpy array.
    arr = clusters_xr.values  # shape: (T, LAT, LON)
    
    # Create a mask for valid points: ignore NaNs and noise (-1).
    valid_mask = (~np.isnan(arr)) & (arr != -1)
    
    # Get the (t, lat, lon) indices for each valid point.
    t_idx, lat_idx, lon_idx = np.where(valid_mask)
    X = np.column_stack((t_idx, lat_idx, lon_idx))
    
    # Get the cluster labels for these points.
    labels = arr[valid_mask].astype(int)
    
    # Check if at least 2 clusters are present.
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        silhouette = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan
    else:
        silhouette = silhouette_score(X, labels, metric='euclidean')
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
    
    # Compute noise ratio: proportion of points labeled as noise (-1) compared to all the non NaN points.
    noise_ratio = np.sum(arr == -1) / np.sum(~np.isnan(arr)) 
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'noise_ratio': noise_ratio
    }