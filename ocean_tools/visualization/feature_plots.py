# ocean_tools/visualization/difference_plots.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def plot_feature_histograms(df, n_rows, n_cols, i_width, i_height, title):
    """
    Plots a grid of stacked histograms (one per feature) from a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame where rows correspond to clusters and columns correspond to features.
    n_rows : int
        Number of rows in the grid of subplots.
    n_cols : int
        Number of columns in the grid of subplots.
    i_width : int or float
        Figure width (in inches).
    i_height : int or float
        Figure height (in inches).
    title : str
        Overall title for the figure.
    """
    # Determine the number of clusters from the DataFrame index.
    n_clusters = len(df.index)
    
    # Generate a list of distinct colors.
    colors = list(mcolors.CSS4_COLORS.values())
    np.random.seed(10)  # For reproducibility
    np.random.shuffle(colors)
    distinct_colors = colors[:n_clusters]
    
    # Create the figure and subplots grid.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(i_width, i_height))
    
    # Flatten the axes array for easy iteration.
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    features = df.columns
    n_features = len(features)
    
    # Loop over each feature and plot its stacked histogram.
    for i, feature_name in enumerate(features):
        ax = axes[i]
        feature_values = df[feature_name].values  # one value per cluster
        # Prepare data as a list of one-value arrays (one per cluster).
        data_list = [[v] for v in feature_values]
        
        # Define uniform bins based on the overall min and max for the feature.
        n_bins = 10
        min_val = np.nanmin(feature_values)
        max_val = np.nanmax(feature_values)
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Plot the stacked histogram.
        ax.hist(data_list, bins=bins, stacked=True, color=distinct_colors, edgecolor='black')
        ax.set_title(f'{feature_name}')
        ax.set_xlim(min_val, max_val)
        
    
    # Remove any unused subplots.
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])
    
    # Create legend handles: map each cluster id (from df.index) to its color.
    legend_handles = [
        mpatches.Patch(color=distinct_colors[i], label=f'Cluster {df.index[i]}')
        for i in range(n_clusters)
    ]
    
    # Add a global legend to the figure at the right side, centered vertically.
    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.85, 0.5))
    
    # Set the overall title and adjust layout.
    fig.suptitle(title, fontsize=16)
    # Adjust layout so that the subplots don't extend into the legend area.
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def plot_feature_scatter_grid(df, i_width, i_height, title):
    """
    Plots a scatterplot matrix (grid) comparing each feature with every other feature.
    Each point represents a cluster (with one row per cluster in df) and is colored according
    to its cluster id. The diagonal shows the feature name.
    
    The left-most column shows y-axis tick marks (min and max values) for each row's feature,
    and the bottom row shows x-axis tick marks (min and max values) for each column's feature.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with clusters as rows and features as columns.
    i_width : int or float
        The overall figure width in inches.
    i_height : int or float
        The overall figure height in inches.
    title : str
        The overall title of the figure.
    """
    features = df.columns
    n_features = len(features)
    # The cluster IDs come from the DataFrame index.
    cluster_ids = df.index.tolist()
    n_clusters = len(cluster_ids)
    
    # Generate distinct colors for each cluster.
    colors = list(mcolors.CSS4_COLORS.values())
    np.random.seed(10)  # For reproducibility
    np.random.shuffle(colors)
    distinct_colors = colors[:n_clusters]
    
    # Compute the min and max for each feature (for tick labeling).
    feature_ranges = {feat: (df[feat].min(), df[feat].max()) for feat in features}
    
    # Create a grid of subplots with shared axes.
    fig, axes = plt.subplots(n_features, n_features, figsize=(i_width, i_height), 
                             sharex='col', sharey='row')
    
    # Ensure axes is 2D.
    if n_features == 1:
        axes = np.array([[axes]])
    else:
        axes = np.array(axes)
    
    # Loop over rows and columns of the grid.
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i == j:
                # Diagonal: display the feature name.
                ax.annotate(features[i], xy=(0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                # Hide the spines.
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                # Off-diagonals: scatter plot comparing feature j (x) vs. feature i (y).
                for k, cluster_label in enumerate(cluster_ids):
                    x_val = df.loc[cluster_label, features[j]]
                    y_val = df.loc[cluster_label, features[i]]
                    ax.scatter(x_val, y_val, color=distinct_colors[k], s=40)
                # Let tick labels be controlled by shared axes.
                ax.tick_params(labelsize=8)
    
    # For the bottom row, set x-axis ticks and label.
    for j in range(n_features):
        ax = axes[-1, j]
        ax.tick_params(axis='x', labelbottom=True)
        mn, mx = feature_ranges[features[j]]
        ax.set_xticks([mn, mx])
        ax.set_xlabel(features[j], fontsize=10)
    
    # For the left column, set y-axis ticks and label.
    for i in range(n_features):
        ax = axes[i, 0]
        ax.tick_params(axis='y', labelleft=True)
        mn, mx = feature_ranges[features[i]]
        ax.set_yticks([mn, mx])
        ax.set_ylabel(features[i], fontsize=10)
    
    # Create legend handles: one patch per cluster.
    legend_handles = [
        mpatches.Patch(color=distinct_colors[k], label=f'Cluster {cluster_ids[k]}')
        for k in range(n_clusters)
    ]
    # Place a global legend at the right, centered vertically.
    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.85, 0.5))
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()