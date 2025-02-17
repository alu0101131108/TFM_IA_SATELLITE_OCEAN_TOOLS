# ocean_tools/visualization/reports.py

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import numpy as np

def plot_n_eof_report(patterns, time_series, max_mins, lat, lon, time_dim, 
                      clim_pattern=[-0.01, 0.01], clim_maxmin=[-5, 5], 
                      cmap='RdBu_r', pat_norm_mode='none', maxmin_norm_mode='none', timeseries_type='line'):
    """
    Genera un reporte con la info de cada EOF:
     - Patrón espacial
     - Serie temporal
     - Campo espacial en el máximo y mínimo de la PC
    """
    n_patterns = len(patterns)
    fig = plt.figure(figsize=(20, n_patterns * 3))
    gs = gridspec.GridSpec(n_patterns, 4, width_ratios=[1, 3, 1, 1])
    
    for i in range(n_patterns):
        # -- EOF pattern
        ax = plt.subplot(gs[i, 0], projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])

        if pat_norm_mode == 'none':
            plt.pcolormesh(lon, lat, patterns[i], cmap=cmap)
            plt.clim(clim_pattern)
        elif pat_norm_mode == 'log':
            plt.pcolormesh(lon, lat, patterns[i], cmap=cmap, 
                           norm=colors.LogNorm(vmin=clim_pattern[0], vmax=clim_pattern[1]))
        elif pat_norm_mode == 'symlog':
            plt.pcolormesh(lon, lat, patterns[i], cmap=cmap,
                           norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1, 
                                                  vmin=clim_pattern[0], vmax=clim_pattern[1]))
        plt.colorbar(label='')
        plt.title(f'EOF {i+1}')

        # -- Time series
        if timeseries_type == 'enhanced_bars':
            ax = plt.subplot(gs[i, 1])
            # Create vertical bar chart with blue bars for positive and red for negative values
            bar_colors = ['blue' if val >= 0 else 'red' for val in time_series[i]]
            plt.bar(time_dim, time_series[i], color=bar_colors, width=np.timedelta64(15, 'D'))
            plt.ylabel('Intensity')
            plt.ylim(-5, 5)
            plt.title(f'TS {i+1}')
            min_year = int(np.datetime_as_string(time_dim[0], unit='Y'))
            max_year = int(np.datetime_as_string(time_dim[-1], unit='Y'))
            for year_line in range(min_year, max_year + 2):
                plt.axvline(x=np.datetime64(f'{year_line}-01-01'), color='black', linestyle='--', linewidth=0.5)
        else:
            ax = plt.subplot(gs[i, 1])
            plt.plot(time_dim, time_series[i])
            plt.ylabel('Intensity')
            plt.ylim(-5, 5)
            plt.title(f'TS {i+1}')
            min_year = int(np.datetime_as_string(time_dim[0], unit='Y'))
            max_year = int(np.datetime_as_string(time_dim[-1], unit='Y'))
            for year_line in range(min_year, max_year + 2):
                plt.axvline(x=np.datetime64(f'{year_line}-01-01'), color='black', linestyle='--', linewidth=0.5)

        # -- Max date
        ax = plt.subplot(gs[i, 2], projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])

        if maxmin_norm_mode == 'none':
            plt.pcolormesh(lon, lat, max_mins[i][0], cmap=cmap)
            plt.clim(clim_maxmin)
        elif maxmin_norm_mode == 'log':
            plt.pcolormesh(lon, lat, max_mins[i][0], cmap=cmap, 
                   norm=colors.LogNorm(vmin=clim_maxmin[0], vmax=clim_maxmin[1]))
        elif maxmin_norm_mode == 'symlog':
            plt.pcolormesh(lon, lat, max_mins[i][0], cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1,
                              vmin=clim_maxmin[0], vmax=clim_maxmin[1]))
        plt.colorbar(label='')
        max_date = np.datetime_as_string(time_dim[np.argmax(time_series[i])], unit='M')
        plt.title(f'Max {i+1} - {max_date}')

        # -- Min date
        ax = plt.subplot(gs[i, 3], projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])
        
        if maxmin_norm_mode == 'none':
            plt.pcolormesh(lon, lat, max_mins[i][1], cmap=cmap)
            plt.clim(clim_maxmin)
        elif maxmin_norm_mode == 'log':
            plt.pcolormesh(lon, lat, max_mins[i][1], cmap=cmap, 
                   norm=colors.LogNorm(vmin=clim_maxmin[0], vmax=clim_maxmin[1]))
        elif maxmin_norm_mode == 'symlog':
            plt.pcolormesh(lon, lat, max_mins[i][1], cmap=cmap,
                   norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1,
                              vmin=clim_maxmin[0], vmax=clim_maxmin[1]))
        plt.colorbar(label='')
        min_date = np.datetime_as_string(time_dim[np.argmin(time_series[i])], unit='M')
        plt.title(f'Min {i+1} - {min_date}')
        
    plt.tight_layout()
    plt.show()


from .feature_plots import plot_feature_histograms, plot_feature_scatter_grid
import pandas as pd
def plot_n_clustering_experiment_feature_report(features):
    # set options
    pd.set_option('display.max_rows', None)           # Show all rows
    pd.set_option('display.max_columns', None)        # Show all columns
    pd.set_option('display.expand_frame_repr', False)   # Prevent line-wrapping
    
    # Table of features per cluster.
    df = pd.DataFrame(features).transpose()
    print(df.to_string())
    # Plot histograms of features.
    plot_feature_histograms(df, n_rows=2, n_cols=4, i_width=10, i_height=5, title="Cluster Feature Histograms")
    # Plot scatterplot matrix of features.
    plot_feature_scatter_grid(df, i_width=25, i_height=20, title="Scatterplot Matrix of Cluster Features")

    # reset options to default
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.expand_frame_repr')