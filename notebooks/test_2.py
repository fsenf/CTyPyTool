from netCDF4 import Dataset 
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy


data_directory = "../data/"
#filename = "msevi_georef.nc"
#filename = "nwcsaf_msevi-nawdex-20160930.nc"
filename = "msevi-nawdex-20160925.nc"
mask_filename = ""


sat_data = Dataset(data_directory + filename)
#print(sat_data)
#print(sat_data.variables)

lons = sat_data.variables['lon']

lats = sat_data.variables['lat']

bt120 = sat_data.variables['bt120']


lons_flat = lons[0]
for row in lons[1:]:
	lons_flat = np.ma.append(lons_flat, row)

lats_flat = lats[0]
for row in lats[1:]:
	lats_flat = np.ma.append(lats_flat, row)

bt120_flat = bt120[0][0]
for row in bt120[0][1:]:
	bt120_flat = np.ma.append(bt120_flat, row)

lon_0 = lons_flat.mean()
lat_0 = lats_flat.mean()



def plotData():

	central_lon = lons_flat.mean()
	central_lat = lats_flat.mean()
	extent = [lons_flat.min(), lons_flat.max(), lats_flat.min(), lats_flat.max()]
	ax = plt.axes(projection=ccrs.PlateCarree())
	ax.set_extent(extent, crs=ccrs.PlateCarree())
	ax.gridlines()
	ax.coastlines(resolution='110m')
	#ax.add_feature(cartopy.feature.OCEAN)
	#ax.add_feature(cartopy.feature.LAND, edgecolor='black')
	#ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
	#ax.add_feature(cartopy.feature.RIVERS)	
	ax.contourf(lons_flat, lats_flat, bt120_flat, cmap = "jet")
	plt.show()

plotData()