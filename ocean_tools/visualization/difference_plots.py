# ocean_tools/visualization/difference_plots.py

import numpy as np
import matplotlib.pyplot as plt

def plot_total_average_differences(diffs, labels, i_width=10, i_height=5, plot_title="Average Differences"):
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