import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

        
def plot_data(data, lons, lats):
    '''
    Plots a satelite-dataset onto a worldmap    
    
    
    Parameters
    ----------
    data : xr.array
        2D-array containig the datapoints 
    
    lons : xr.array
        2D-array of the longitude values for each datapoint

    lats : xr.array
        2D-array of the lattidude values for each datapoint
    
    '''
    plt.figure(figsize=(12, 4))
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.gridlines()
    ax.coastlines(resolution='50m')
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.RIVERS)  
    ax.contourf(lons, lats, data, cmap = "jet")
    plt.show()


def addSubplot(data, variable = None, lons = None, lats = None, time = 0, index = 0, size_x = 1, size_y = 1):
    if lons is None or lats is None:
        try:
            lons = data.variables['lon']
            lats = data.variables['lat']
        except Exception:
            print("Longitude/Lattide variables not found!")
            return
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    ax = plt.subplot(size_x,size_y,index, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.RIVERS)  
    if variable == None:
        ax.contourf(lons, lats, data ,cmap = "jet")
    else:
        ax.contourf(lons, lats, data[variable][time],cmap = "jet")

        