import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
import shapely
import xarray as xr

def plot_data(data, indices = None, hour = None):
    '''
    Plots labels from an xarray onto a worldmap    
    
    
    Parameters
    ----------
    data : xr.array
        2D-array containig the datapoints 
    
    lons : xr.array
        2D-array of the longitude values for each datapoint

    lats : xr.array
        2D-array of the lattidude values for each datapoint
    
    '''
    shapely.speedups.disable()
    try:
        lons = data.coords['lon']
        lats = data.coords['lat']
    except Exception:
        print("Longitude/Lattide variables not found!")
        return

    if (hour is None):
        data = data["CT"]
    else:
        data = data["CT"][hour]



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


    cmap = plt.matplotlib.colors.ListedColormap( ct_colors )

    # shrink to area
    new_data = np.empty(data.shape)
    new_data[:] = np.nan
    if (indices is None):
        indices = np.where(~np.isnan(data))
    data = np.array(data)[indices[0], indices[1]]
    new_data[indices[0],indices[1]] = data
    #new_data = xr.DataArray(new_data)


    #lons = lons[~np.isnan(data)]
    #lats = lats[~np.isnan(data)]
    #extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    plt.figure(figsize=(12, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    #ax.set_extent(extent)
    ax.gridlines()
    ax.coastlines(resolution='50m')
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.RIVERS)  
    ax.contourf(lons, lats, new_data, cmap = cmap)
    plt.show()


# def addSubplot(data, variable = None, lons = None, lats = None, time = 0, index = 0, size_x = 1, size_y = 1):
#     if lons is None or lats is None:
#         try:
#             lons = data.variables['lon']
#             lats = data.variables['lat']
#         except Exception:
#             print("Longitude/Lattide variables not found!")
#             return
#     extent = [lons.min(), lons.max(), lats.min(), lats.max()]
#     ax = plt.subplot(size_x,size_y,index, projection=ccrs.PlateCarree())
#     ax.set_extent(extent, crs=ccrs.PlateCarree())
#     ax.coastlines(resolution='50m')
#     ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
#     ax.add_feature(cartopy.feature.RIVERS)  
#     if variable == None:
#         ax.contourf(lons, lats, data ,cmap = "jet")
#     else:
#         ax.contourf(lons, lats, data[variable][time],cmap = "jet")

#         