import torch
import json
import argparse
import netCDF4
from models.newcae2 import YourNet
from utils.utils import save_tr_code
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='C:/AEcode/configs/config1.json', default="./configs/config1.json")
args = parser.parse_args()
config = args.config_path


args = json.load(open(args.config_path))
save_tr_code(args['save_dir'], config, 'test_era.py')

device = torch.device(args['device'] if torch.cuda.is_available() else "cpu")
print('\t>>> Device used: {}'.format(device))

model = YourNet(input_channels=1, use_masklayer=args['use_masklayer'], masklayer_code_sizes=args['masklayer_code_sizes']
                ).to(device)

print('\t>>> Use pretrain model.')
checkpoint = torch.load(args['pretrain'])
start_epoch = checkpoint['epoch'] + 1
model.load_state_dict(checkpoint['model_state_dict'])
print('\t>>> Pretrain model loaded.')

#file = 'C:/AEcode/data/era20anomal_fin.nc'
#file = 'C:/AEcode/anomal_z500_set1.nc'
#file = 'C:/AEcode/anomal_z500_set2.nc'
file = 'C:/AEcode/anomal_z500_set3.nc'
d = netCDF4.Dataset(file, mode='r')

#datas = d['z500'][:] #uncomment for era20anomal_fin
datas = d['zg500'][:]

#datas = normalize(datas)
#dataset = datas[0:32434] #actually don't need this for clustering
#test_dataset = datas[32434:] #uncomment for era20
test_dataset = datas[:]
print(np.min(datas), np.max(datas))

start = 0
end = 0
encods2 = np.empty((0, 60))
b = args['batch_size']
num_test = len(test_dataset)
print(num_test)
for i in range(num_test // b):
    # start and end to indicate the batch indexes
    end = start + b
    if end >= num_test:
        end = num_test
    # data_t will save the one batch values from start to end
    data_t = test_dataset[start:end, :]
    data_t = torch.tensor(data_t)
    start = 0 if end >= num_test else end
    #data_t = torch.tensor(data_t)
    data_t = data_t.to(device)
    # preds2 takes the decoded ouput of the model
    preds2 =model(data_t.unsqueeze(1), use_masklayer=args['test_use_masklayer'], masklayer_code_sizes=args['test_code_sizes'])[0]
    # latent2 takes the encoded output of the model
    latent2 =model(data_t.unsqueeze(1), use_masklayer=args['test_use_masklayer'], masklayer_code_sizes=args['test_code_sizes'])[1]
    latent2 = latent2.cpu().detach().numpy()
# collect all encodings
    encods2 = np.vstack((encods2, latent2))


print("Encodings size" + str(encods2.shape))


#K means
kmeans = KMeans(n_clusters=8, random_state=0).fit(encods2)
#print(kmeans.labels_)
print(kmeans.labels_.shape)
#print(kmeans.cluster_centers_)

#visualise
clus = []  # to have all the clusters
distr = []
for label in set(kmeans.labels_):
    cluster = []
    count = 0
    for i in range(len(kmeans.labels_)):
        if (kmeans.labels_[i] == label):
            cluster.append(encods2[i])
            count += 1
    distr.append(count)
    clus.append(np.average(np.vstack(cluster), axis=0))
print(distr)


# lats = d.variables['latitude'] #for era
# longs = d.variables['longitude']  #for era
lats = d.variables['lat'] #for cmip
longs = d.variables['lon']  #for cmip

longs, lats = np.meshgrid(longs, lats)


final=[]
clustxt = []

for i in range(len(clus)):

    test = torch.unsqueeze(torch.tensor(kmeans.cluster_centers_[i], dtype=torch.float32).to(device), 0) #torch.tensor(clus[i]).to(device)
    out = model(test, mode=True)[0] #why mode
    final.append(out)

m = Basemap(projection='merc',
                llcrnrlon=-58,  # lower longitude
                llcrnrlat=25,  # lower latitude
                urcrnrlon=32,  # uppper longitude
                urcrnrlat=70,  # uppper latitude
                resolution='i')
xi, yi = m(longs, lats)
#to save the decoded cluster centers
arrtxt = []

for i in range(0, 8):
    # Plot Data
    plt.figure()


    inp2 = np.reshape(final[i].cpu().detach().numpy(), (32, 64))

    arrtxt.append(inp2.flatten())

    cs = m.pcolor(xi, yi, inp2, shading='nearest', cmap='coolwarm')  # , vmin=np.min(test_dataset), vmax=np.max(test_dataset)

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
    plt.title('CMIP_SET3_Cluster ' + str(i + 1))
    plt.savefig('CMIP_SET3_cluster_lat8_%03d.png' % (i+1))

np.savetxt("CMIP_SET3_clusters_latent.txt", arrtxt)
    #plt.show()

