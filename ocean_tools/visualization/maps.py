# ocean_tools/visualization/maps.py

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors

def plot_spatial_variable(target, lat, lon, i_width=8, i_height=8, 
                          clim=[0, 40], title="", plot_label='Variable', 
                          cmap='RdBu_r', norm_mode='none'):
    """
    Grafica una sola matriz 2D (lat, lon) en un mapa usando cartopy.
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
    Grafica múltiples matrices 2D (lat, lon) en subplots.
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
