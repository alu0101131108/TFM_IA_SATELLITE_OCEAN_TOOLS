# ocean_tools/visualization/difference_plots.py

import numpy as np
import matplotlib.pyplot as plt

def plot_total_average_differences(diffs, labels, i_width=10, i_height=5, plot_title="Average Differences"):
    """
    Plots overall average differences as a horizontal bar chart.

    Parameters
    ----------
    * diffs : list or array-like
        * A list or array of scalar difference values.
    * labels : list or array-like
        * A list of labels corresponding to each difference (e.g., season names or month numbers).
    * i_width : int, optional
        * Width of the figure in inches (default is 10).
    * i_height : int, optional
        * Height of the figure in inches (default is 5).
    * plot_title : str, optional
        * Title of the plot (default is "Average Differences").

    Returns
    -------
    None
    """
    plt.figure(figsize=(i_width, i_height))
    keys = labels
    values = diffs
    bars = plt.barh(keys, values, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    plt.title(plot_title)
    plt.grid(axis='x')
    # Add labels at end of each bar.
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center')
    plt.show()

def plot_latitudinal_average_diffences(diffs, labels, lat=None, smooth_factor=0, i_width=5, i_height=5, plot_label=None, plot_title="Latitudinal Average Differences"):
    """
    Plots latitudinal average differences as line charts.

    For each set of difference values (assumed to be 1D arrays), this function optionally applies
    a moving average smoothing (if smooth_factor >= 1) and plots the resulting line with the corresponding label.
    A vertical line at x=0 is drawn for reference.

    Parameters
    ----------
    * diffs : list
        * List of 1D arrays representing average differences (e.g., differences averaged over longitude).
    * labels : list
        * List of labels for each profile (e.g., season names).
    * lat : np.ndarray, optional
        * 1D array of latitude values. Required if diffs do not include coordinate information.
    * smooth_factor : float, optional
        * Smoothing factor in the same units as latitude. If >= 1, moving average smoothing is applied.
        * Default is 0 (no smoothing).
    * i_width : int, optional
        * Width of the figure in inches (default is 5).
    * i_height : int, optional
        * Height of the figure in inches (default is 5).
    * plot_label : str, optional
        * Label for the x-axis. If provided, used as the x-axis label.
    * plot_title : str, optional
        * Title of the plot (default is "Latitudinal Average Differences").

    Returns
    -------
    None
    """
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = int(smooth_factor / 0.041)  # 0.041 is the lat step size. This window size will smooth over lat_smooth degrees.
    
    plt.figure(figsize=(i_width, i_height))

    n_diffs = len(diffs)
    for i in range(n_diffs):
        if smooth_factor >= 1:
            diff_mean_smooth = moving_average(diffs[i], window_size)
            lat_smooth = lat[window_size-1:]
            plt.plot(diff_mean_smooth, lat_smooth, label=labels[i], linewidth=1)
        else:
            plt.plot(diffs[i], lat, label=labels[i], linewidth=1)

    # Add a vertical line at x=0
    plt.axvline(x=0, color='r', linestyle='--', linewidth=3)

    # Add labels and title
    plt.ylabel('Latitude')
    if plot_label: plt.xlabel(plot_label)
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_longitudinal_average_diffences(diffs, labels, lon=None, smooth_factor=0, i_width=5, i_height=5, plot_label=None, plot_title="Latitudinal Average Differences"):
    """
    Plots longitudinal average differences as line charts.

    For each set of difference values (assumed to be 1D arrays), this function optionally applies
    a moving average smoothing (if smooth_factor >= 1) and plots the resulting line with the corresponding label.
    A horizontal line at y=0 is drawn for reference.

    Parameters
    ----------
    * diffs : list
        * List of 1D arrays representing average differences (e.g., differences averaged over latitude).
    * labels : list
        * List of labels for each profile (e.g., season names or month numbers).
    * lon : np.ndarray, optional
        * 1D array of longitude values. Required if diffs do not include coordinate information.
    * smooth_factor : float, optional
        * Smoothing factor in the same units as longitude. If >= 1, moving average smoothing is applied.
        * Default is 0 (no smoothing).
    * i_width : int, optional
        * Width of the figure in inches (default is 5).
    * i_height : int, optional
        * Height of the figure in inches (default is 5).
    * plot_label : str, optional
        * Label for the y-axis. If provided, used as the y-axis label.
    * plot_title : str, optional
        * Title of the plot (default is "Longitudinal Average Differences").

    Returns
    -------
    None
    """
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = int(smooth_factor / 0.041)  # 0.041 is the lon step size. This window size will smooth over lon_smooth degrees. (~60 pixels)
    
    plt.figure(figsize=(i_width, i_height))

    n_diffs = len(diffs)
    for i in range(n_diffs):
        if smooth_factor >= 1:
            diff_mean_smooth = moving_average(diffs[i], window_size)
            lon_smooth = lon[window_size-1:]
            plt.plot(lon_smooth, diff_mean_smooth, label=labels[i], linewidth=1)
        else:
            plt.plot(lon, diffs[i], label=labels[i], linewidth=1)

    # Add a horizontal line at x=0
    plt.axhline(y=0, color='r', linestyle='--', linewidth=3)

    # Add labels and title
    plt.xlabel('Longitude')
    if plot_label: plt.ylabel(plot_label)
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.show()