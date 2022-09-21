import torch
import json
import argparse
import netCDF4
from models.your_net import YourNet
from utils.utils import save_tr_code
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='C:/AEcode/configs/config1.json', default="./configs/config1.json")
args = parser.parse_args()
config = args.config_path


args = json.load(open(args.config_path))
save_tr_code(args['save_dir'], config, 'test.py')

device = torch.device(args['device'] if torch.cuda.is_available() else "cpu")
print('\t>>> Device used: {}'.format(device))

model = YourNet(input_channels=1, use_masklayer=args['use_masklayer'], masklayer_code_sizes=args['masklayer_code_sizes']
                ).to(device)

print('\t>>> Use pretrain model.')
checkpoint = torch.load(args['pretrain'])
start_epoch = checkpoint['epoch'] + 1
model.load_state_dict(checkpoint['model_state_dict'])
print('\t>>> Pretrain model loaded.')

file = 'C:/AEcode/mslp.nc'
d = netCDF4.Dataset(file, mode='r')
test_dataset = d['mslp_anom'][44997:]
dataset = d['mslp_anom'][0:44997]

b = args['batch_size']
writer = SummaryWriter(args["save_dir"])
num_test = len(test_dataset)
all_losses = []
loss_function = nn.MSELoss()
for j in args['masklayer_code_sizes']:
    start = 0
    end = 0
    tr_loss = 0
    for i in range(num_test // b):

        end = start + b
        if end >= num_test:
            end = num_test
        data = test_dataset[start:end, :]
        data = torch.tensor(data)
        start = 0 if end >= num_test else end
        data = data.to(device)
        # print(data.size())
        testing = data.unsqueeze(1)

        preds1 = model(data.unsqueeze(1), use_masklayer=args['test_use_masklayer'], masklayer_code_sizes=[j])[0]

        loss = loss_function(preds1, data.unsqueeze(1))
        tr_loss += loss.item()
    size_tr_loss = tr_loss / (num_test // b)
    writer.add_scalar('Test_loss', size_tr_loss, j)
    #print(epoch_tr_loss,j)
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
    #layer1 = np.loadtxt("size_40_23.txt", delimiter=',', dtype=float)
    #layer2 = np.loadtxt("size_50_23.txt", delimiter=',', dtype=float)
    # inp2 = np.subtract(layer2, layer1)
    # for output
    test = test_dataset[0:100]
    test = torch.tensor(test)
    test = test.to(device)
    # print(test[11])
    tout = model(test.unsqueeze(1), use_masklayer=args['test_use_masklayer'], masklayer_code_sizes=[j])[0]
    loss = loss_function(tout, test.unsqueeze(1))
    print("Size{} Loss: {}".format(j,loss.item()))
    all_losses.append(loss.item())
    inp2 = np.reshape(tout[11].cpu().detach().numpy(), (10, 25))
    #save the result in txt format for subtractLayers.py
    result = np.savetxt('size_%.txt', inp2, delimiter=',')
    np.savetxt(('MaskL_size_12_50_'+str(j)+'.txt'),inp2,delimiter=',')
    # inp2 = np.reshape(test_dataset[23], (10, 25))
    cs = m.pcolor(xi, yi, inp2, shading='nearest', cmap='coolwarm')  #vmin=np.min(test_dataset) vmax=np.max(test_dataset)

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
    plt.title('Mask Layer [%03d]'%(j))
    plt.savefig('MaskLayer_14_%03d.png' % (j))

    #plt.show()


print("MSE loss: "+str(loss.item()))
print(all_losses)



