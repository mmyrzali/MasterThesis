import torch
import netCDF4
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt

file = 'C:/AEcode/mslp.nc'
d = netCDF4.Dataset(file, mode='r')
test_dataset = d['mslp_anom'][44997:]
dataset = d['mslp_anom'][0:44997]

# Visualize the map with mask layer showing only the first X features
lats = d.variables['lat']
longs = d.variables['lon']
longs, lats = np.meshgrid(longs, lats)
m = Basemap(projection='merc',
            llcrnrlon=-70,  # lower longitude
            llcrnrlat=25,  # lower latitude
            urcrnrlon=50,  # uppper longitude
            urcrnrlat=70,  # uppper latitude
            resolution='i')
xi, yi = m(longs, lats)

plt.figure()
layer1 = np.loadtxt("MaskL_size_12_50_20.txt", delimiter=',', dtype=float)
layer2 = np.loadtxt("MaskL_size_12_50_30.txt", delimiter=',', dtype=float)
inp2 = np.subtract(layer2, layer1)
#inp2 = layer1
# for output
# inp2 = np.reshape(test_dataset[23], (10, 25))
cs = m.pcolor(xi, yi, inp2, shading='nearest', cmap='coolwarm', vmin=np.min(inp2),
              vmax=np.max(inp2))  #

# Add Grid Lines
m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Add Colorbar
cbar = m.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('MSLP hPa')

# Add Title
plt.title('Mask Layer [30-20]')
plt.savefig('masklayer_12_lat30_50_30_20.png')

plt.show()
