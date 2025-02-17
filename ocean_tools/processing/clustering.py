# ocean_tools/processing/clustering.py

import xarray as xr
import numpy as np

# Anomaly segmentation into components.
def segment_anomalies(ds_anom, variable_name, anomaly_threshold):
    
    ds_anom_var = abs(ds_anom[variable_name])
    
    # Label relevant anomalies.
    ds_segmented = xr.where((ds_anom_var >= anomaly_threshold) & ds_anom_var.notnull(), 1, np.nan)
    
    return ds_segmented

# Flood-fill like algorithm for connected component labeling.
def label_connected_components(ds_segmented, eps_time=1, eps_lat=1, eps_lon=1, min_cluster_size=5):
    # Convert segmented DataArray to a NumPy array.
    seg_arr = ds_segmented.values
    T, LAT, LON = seg_arr.shape
    
    # Initialize the output label array. Use a float type to allow np.nan.
    labels = np.full(seg_arr.shape, np.nan, dtype=float)
    # Preserve np.nan from the original segmented data.
    nan_mask = np.isnan(seg_arr)
    labels[nan_mask] = np.nan
    
    # Boolean array to keep track of which points have been visited. NaNs are considered visited.
    visited = np.zeros(seg_arr.shape, dtype=bool)
    visited[nan_mask] = True

    # Global label counter for valid clusters.
    current_label = 1
    
    def in_bounds(t, i, j):
        """Helper function to check if indices (t, i, j) are inside the array bounds."""
        return (0 <= t < T) and (0 <= i < LAT) and (0 <= j < LON)

    # Iterate over all points in the 3D array.
    for t in range(T):
        for i in range(LAT):
            for j in range(LON):
                if visited[t, i, j]: continue # Skip visited points.
                
                # Begin flood-fill using a stack and a list to store component points.
                stack = [(t, i, j)]
                component_points = []
                while stack:
                    tt, ii, jj = stack.pop()
                    if not in_bounds(tt, ii, jj) or visited[tt, ii, jj]: continue # Skip out-of-bounds or visited points.

                    # Mark this point as visited and include in the component.
                    visited[tt, ii, jj] = True
                    component_points.append((tt, ii, jj))

                    # Loop over neighbors within the eps windows.
                    for dt in range(-eps_time, eps_time + 1):
                        for di in range(-eps_lat, eps_lat + 1):
                            for dj in range(-eps_lon, eps_lon + 1):
                                nt = tt + dt
                                ni = ii + di
                                nj = jj + dj
                                if not in_bounds(nt, ni, nj) or visited[nt, ni, nj]: continue # Skip out-of-bounds or visited points.
                                
                                # Add to stack
                                stack.append((nt, ni, nj))
                
                # Now that the flood-fill is complete, determine the cluster size.
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
    Performs DBSCAN-like clustering on a segmented xarray DataArray (dimensions: time, lat, lon)
    using custom spatiotemporal connectivity and density requirements.
    
    Relevant points are assumed to have value 1 in ds_segmented (non-relevant are 0 or NaN).
    
    A point is a core point if its eps-neighborhood (indices within ±eps_time, ±eps_lat, ±eps_lon)
    contains at least min_neighbors (including itself). The cluster expansion (flood-fill) proceeds
    through core points. All points reached (including border points) are assigned the same cluster label.
    
    After expansion, if the total number of points in the cluster is below min_cluster_size, then
    the entire cluster is discarded and its points are labeled as noise (-1).
    
    Parameters
    ----------
    ds_segmented : xarray.DataArray
        A DataArray with dimensions (time, lat, lon) where:
          - Relevant points (anomaly above threshold) have value 1.
          - Non-relevant points are 0 or NaN.
    eps_time : int, default=1
        Maximum index difference in the time dimension to consider a neighbor.
    eps_lat : int, default=1
        Maximum index difference in the latitude dimension.
    eps_lon : int, default=1
        Maximum index difference in the longitude dimension.
    min_neighbors : int, default=5
        Minimum number of points (including the point itself) required for a point to be
        considered a core point.
    min_cluster_size : int, default=10
        Minimum number of points for a cluster to be retained; clusters smaller than this will be
        discarded (labeled as noise, i.e. -1).
        
    Returns
    -------
    ds_labels : xarray.DataArray
        A DataArray with the same coordinates and dimensions as ds_segmented, where each cluster is 
        assigned a unique positive integer label and noise points are labeled as -1.
    n_clusters : int
        The number of clusters that meet the min_cluster_size requirement.
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
        Returns a list of (t, i, j) tuples for all points within the spatiotemporal
        eps-neighborhood of (t, i, j) that are relevant (i.e. have value 1).
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
    Given an experiment's cluster features, compute experiment-level aggregated features.
    
    Parameters
    ----------
    exp_feature : dict
        Dictionary with keys:
           - 'experiment_name': experiment name,
           - 'file_name': file name,
           - 'features': dict mapping cluster IDs to cluster feature dictionaries.
    
    Returns
    -------
    metrics : dict
        A dictionary containing experiment-level features, including the number of clusters and,
        for each cluster feature (e.g., 'cluster_size', 'cluster_volume', etc.),
        the mean, standard deviation, min, and max.
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
    Computes several clustering metrics for the clustering result stored in an xarray DataArray.
    
    Parameters
    ----------
    clusters_xr : xarray.DataArray
        Clustering output with dimensions (time, lat, lon). Valid points have a positive integer
        cluster label, noise points are labeled -1, and missing data are NaN.
    
    Returns
    -------
    metrics : dict
        A dictionary containing:
          - 'silhouette': silhouette score (higher is better),
          - 'davies_bouldin': Davies-Bouldin index (lower is better),
          - 'calinski_harabasz': Calinski-Harabasz index (higher is better),
          - 'noise_ratio': proportion of noise points.
        If not computable (e.g., fewer than 2 clusters), values will be np.nan.
    """
    # Extract the underlying numpy array.
    arr = clusters_xr.values  # shape: (T, LAT, LON)
    
    # Create a mask for valid points: ignore NaNs and noise (-1).
    valid_mask = (~np.isnan(arr)) & (arr != -1)
    num_valid = np.sum(valid_mask)
    
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
    
    # Compute noise ratio: proportion of points labeled as noise (-1) in the entire array.
    noise_ratio = np.sum(arr == -1) / np.prod(arr.shape)
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'noise_ratio': noise_ratio
    }