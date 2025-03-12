# ocean_tools/visualization/maps.py

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors

def plot_spatial_variable(target, lat, lon, i_width=8, i_height=8, 
                          clim=[0, 40], title="", plot_label='Variable', 
                          cmap='RdBu_r', norm_mode='none'):
    """
    Plots a single 2D spatial variable on a map using Cartopy.

    The function displays the data as a pcolormesh plot with coastlines and gridlines,
    applying a specified colormap and normalization.

    Parameters
    ----------
    * target : array-like
        * 2D array (lat, lon) of the variable to plot.
    * lat : array-like
        * 1D array of latitude values.
    * lon : array-like
        * 1D array of longitude values.
    * i_width : int, optional
        * Figure width in inches (default is 8).
    * i_height : int, optional
        * Figure height in inches (default is 8).
    * clim : list, optional
        * Color limits as [min, max] (default is [0, 40]).
    * title : str, optional
        * Title of the plot (default is an empty string).
    * plot_label : str, optional
        * Label for the colorbar (default is 'Variable').
    * cmap : str, optional
        * Colormap to use (default is 'RdBu_r').
    * norm_mode : str, optional
        * Normalization mode for color scaling. Options are:
            * 'none': no normalization,
            * 'log': logarithmic normalization,
            * 'symlog': symmetric logarithmic normalization.
        * (Default is 'none'.)

    Returns
    -------
    None
    """
    plt.figure(figsize=(i_width, i_height))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])

    if norm_mode == 'none':
        plt.pcolormesh(lon, lat, target, cmap=cmap)
        plt.clim(clim)
    elif norm_mode == 'log':
        plt.pcolormesh(lon, lat, target, cmap=cmap, norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]))
    elif norm_mode == 'symlog':
        plt.pcolormesh(lon, lat, target, cmap=cmap, 
                       norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1, 
                                              vmin=clim[0], vmax=clim[1]))

    plt.colorbar(label=plot_label)
    plt.title(title, fontsize=10)
    plt.show()


def plot_n_spatial_variable(target_list, lat, lon, nrows, ncols, i_width=8, i_height=8, 
                            clim=[0, 40], titles=[], plot_label='Variable', 
                            cmap='RdBu_r', norm_mode='none'):
    """
    Plots multiple 2D spatial variables in a grid of subplots.

    If the colormap 'labels' is specified, the function creates a ListedColormap with distinct colors.
    Each subplot is plotted using Cartopy with coastlines and gridlines. If titles are provided,
    each subplot is given a title.

    Parameters
    ----------
    * target_list : list
        * List of 2D arrays (lat, lon) to plot.
    * lat : list or array-like
        * 1D array of latitude values.
    * lon : list or array-like
        * 1D array of longitude values.
    * nrows : int
        * Number of rows in the subplot grid.
    * ncols : int
        * Number of columns in the subplot grid.
    * i_width : int, optional
        * Width of each subplot in inches (default is 8).
    * i_height : int, optional
        * Height of each subplot in inches (default is 8).
    * clim : list, optional
        * Color limits as [min, max] (default is [0, 40]).
    * titles : list, optional
        * List of titles for each subplot (default is an empty list).
    * plot_label : str, optional
        * Label for the colorbar (default is 'Variable').
    * cmap : str, optional
        * Colormap to use. If 'labels', a discrete colormap is generated (default is 'RdBu_r').
    * norm_mode : str, optional
        * Normalization mode for color scaling. Options are:
            * 'none': no normalization,
            * 'log': logarithmic normalization,
            * 'symlog': symmetric logarithmic normalization.
        * (Default is 'none'.)

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    if cmap == 'labels':
        n_labels = clim[1] - clim[0] - 1
        color_list = list(mcolors.CSS4_COLORS.values())
        np.random.seed(10)  # For reproducibility
        np.random.shuffle(color_list)
        distinct_colors = color_list[:n_labels]
        cmap = mpl.colors.ListedColormap(['#000000'] + distinct_colors)
    
    plt.figure(figsize=(i_width * ncols, i_height * nrows))
    for i, target in enumerate(target_list):
        ax = plt.subplot(nrows, ncols, i + 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]])

        if norm_mode == 'none':
            plt.pcolormesh(lon, lat, target, cmap=cmap)
            plt.clim(clim)
        elif norm_mode == 'log':
            plt.pcolormesh(lon, lat, target, cmap=cmap, norm=colors.LogNorm(vmin=clim[0], vmax=clim[1]))
        elif norm_mode == 'symlog':
            plt.pcolormesh(lon, lat, target, cmap=cmap, 
                           norm=colors.SymLogNorm(linthresh=0.1, linscale=0.1, 
                                                  vmin=clim[0], vmax=clim[1]))
        plt.colorbar(label=plot_label)

        if len(titles) > i:
            plt.title(titles[i], fontsize=10)

    plt.tight_layout()
    plt.show()
