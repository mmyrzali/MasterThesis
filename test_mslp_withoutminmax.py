import torch
import json
import argparse
import netCDF4
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt

inspace = np.loadtxt("clusters.txt", delimiter=' ', dtype=float)
latent = np.loadtxt("clusters_latent.txt",delimiter=' ', dtype=float)

file = 'C:/AEcode/mslp.nc'
d = netCDF4.Dataset(file, mode='r')
test_dataset = d['mslp_anom'][44997:]
dataset = d['mslp_anom'][0:44997]

lats = d.variables['lat']
longs = d.variables['lon']
longs, lats = np.meshgrid(longs, lats)


for j in range(len(latent)):
    m = Basemap(projection='merc',
                llcrnrlon=-70,  # lower longitude
                llcrnrlat=25,  # lower latitude
                urcrnrlon=50,  # uppper longitude
                urcrnrlat=70,  # uppper latitude
                resolution='i')
    xi, yi = m(longs, lats)

    # Plot Data
    plt.figure()
    inp2 = latent[j]
    inp2 = np.reshape(inp2, (10, 25))
    cs = m.pcolor(xi, yi, inp2, shading='nearest',
                  cmap='coolwarm')  # vmin=np.min(test_dataset),vmax=np.max(test_dataset)

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
    plt.title('Cluster ' + str(j + 1))
    plt.savefig('Latent_cluster_%03d.png' % (j + 1))






