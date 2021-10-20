import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy
import cartopy.crs as ccrs
import numpy as np
import xarray as xr







def definde_NWCSAF_variables():
    ct_colors = ['#007800', '#000000','#fabefa','#dca0dc',
            '#ff6400', '#ffb400', '#f0f000', '#d7d796',
            '#e6e6e6',  '#c800c8','#0050d7', '#00b4e6',
            '#00f0f0', '#5ac8a0', ]

    ct_indices = [ 1.5, 2.5, 3.5, 4.5, 
               5.5, 6.5, 7.5, 8.5, 
               9.5, 10.5, 11.5, 12.5,
               13.5, 14.5, 15.5]

    ct_labels = ['land', 'sea', 'snow', 'sea ice', 
                 'very low', 'low', 'middle', 'high opaque', 
                 'very high opaque', 'fractional', 'semi. thin', 'semi. mod. thick', 
                 'semi. thick', 'semi. above low','semi. above snow']

    return ct_colors, ct_indices, ct_labels

def plot_data(data, x, y):
    """
    Plots labels from an xarray onto a worldmap    
    
    
    Parameters
    ----------
    data : xr.array
        2D-array containig the datapoints 
    """

    ct_colors, ct_indices, ct_labels = definde_NWCSAF_variables()


    extent = [-8, 45, 30, 45]
    plt.figure(figsize=(12, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.set_extent(extent)
    #ax.gridlines()
    # ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    # ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    # ax.add_feature(cartopy.feature.RIVERS)  

    cmap = plt.matplotlib.colors.ListedColormap( ct_colors )
    pcm = ax.pcolormesh(x,y, data, cmap = cmap, vmin = 1, vmax = 15)
    

    fig = plt.gcf()
    a2 = fig.add_axes( [0.95, 0.22, 0.015, 0.6])     
    cbar = plt.colorbar(pcm, a2)
    cbar.set_ticks(ct_indices)
    cbar.set_ticklabels(ct_labels)
    plt.show()



def plot_multiple(results, x,y, ground_truth = None, result_types = None, timestamp = None):
    
    ct_colors, ct_indices, ct_labels = definde_NWCSAF_variables()
    cmap = plt.matplotlib.colors.ListedColormap(ct_colors)

    extent = [-6, 42, 25, 50]

    if(not ground_truth is None):
        results.append(ground_truth)
    length = len(results)

    plt.figure(figsize=(12, 4))
    for i in range(length):
        ax = plt.subplot(1,length,i+1, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black')
        ax.coastlines(resolution='50m')
        ax.set_extent(extent)
        pcm = ax.pcolormesh(x,y, results[i], cmap = cmap, vmin = 1, vmax = 15)



