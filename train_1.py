'''
MaskLayer train.py
Written by Remco Royen
'''

import os
import json
import argparse

import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
import netCDF4
from models.your_net import YourNet
from utils.utils import save_tr_code
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
import torchvision
#from torchmetrics.functional import mean_absolute_percentage_error as mape

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='C:/AEcode/configs/config1.json', default="./configs/config1.json")
args = parser.parse_args()
config = args.config_path

print('Initialization')
print('\t>>> Loading {}'.format(args.config_path))

args = json.load(open(args.config_path))
save_tr_code(args['save_dir'], config, 'train_1.py')

device = torch.device(args['device'] if torch.cuda.is_available() else "cpu")
print('\t>>> Device used: {}'.format(device))

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


file = 'C:/AEcode/mslp.nc'
d= netCDF4.Dataset(file, mode='r')
data_all = d['mslp_anom'][:]
minx = np.min(data_all)
maxx = np.max(data_all)
data_all = normalize(data_all)
dataset = data_all[0:44997]
test_dataset = data_all[44997:]

def denormalize(x):
    return x*(maxx-minx)+minx

print('Size of train data: '+str(len(dataset)))
print('Size of test data: '+str(len(test_dataset)))


def mape(y_true, y_pred):
    actual, pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100




model = YourNet(input_channels=1, use_masklayer=args['use_masklayer'], masklayer_code_sizes=args['masklayer_code_sizes']
                ).to(device)

loss_function = nn.MSELoss()

if args['optimizer'] == 'Adam':
    optimizer = opt.Adam(model.parameters(), lr=args['lr'])
elif args['optimizer'] == 'SGD':
    optimizer = opt.SGD(model.parameters(), momentum=args['momentum'], lr=args['lr'],
                        weight_decay=args['weight_decay'])
else:
    raise NotImplementedError('The specified optimizer is not (yet) implemented. Do it now ... quick!')

if args['pretrain'] is False:
    print('\t>>> No existing model, starting training from scratch.')
    start_epoch = 0
else:
    print('\t>>> Use pretrain model.')
    checkpoint = torch.load(args['pretrain'])
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('\t>>> Pretrain model loaded.')

writer = SummaryWriter(args["save_dir"])

print('Training')

optimizer.zero_grad()
best_value = np.inf
start = 0
end = 0
b = args['batch_size']
num_samples = len(dataset)
num_test = len(test_dataset)
for epoch in range(start_epoch, args['epochs']+1):
    print('\t>>> Epoch: %d' % epoch)

    tr_loss = 0
    for i in range(num_samples//b):
        end = start + b
        if end >= num_samples:
            end = num_samples
        data = dataset[start:end, :]
        data = torch.tensor(data)
        start = 0 if end >= num_samples else end
        data = data.to(device)
        # print(data.size())
        testing = data.unsqueeze(1)
        #print(testing.size())
        preds1 = model(data.unsqueeze(1),use_masklayer=args['test_use_masklayer'], masklayer_code_sizes=args['test_code_sizes'])[0]
        #print(preds1)
        #print(preds1.size())
        loss = loss_function(preds1, data.unsqueeze(1)) #mape(preds1, data.unsqueeze(1)) #
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar('Batch_tr_loss', loss.item()/b, i + num_samples//b*epoch)

        tr_loss += loss.item()

    epoch_tr_loss = tr_loss /(num_samples//b) #loader.
    writer.add_scalar('Epoch_tr_loss', epoch_tr_loss, epoch)
    print('\t\t>>> Epoch {} finished: epoch_tr_loss: {}'.format(epoch, epoch_tr_loss))

    model.eval()
    test_loss = 0
    start = 0
    end=0
    for i in range(num_test//b):
        #data = test_dataset[i]
        #data = data.reshape(-1, 250)
        end = start + b
        if end >= num_test:
            end = num_test
        data_t = test_dataset[start:end, :]
        #data_t = data_t[:, :, :22]
        data_t = torch.tensor(data_t)
        start = 0 if end >= num_test else end
        data_t = torch.tensor(data_t)
        data_t = data_t.to(device)
        preds2 = model(data_t.unsqueeze(1), use_masklayer=args['test_use_masklayer'], masklayer_code_sizes=args['test_code_sizes'])[0]
        loss = loss_function(preds2, data_t.unsqueeze(1)) #mape(preds2, data_t.unsqueeze(1))#
        writer.add_scalar('Batch_test_loss', loss.item()/b, i + num_test//b*epoch)
        test_loss += loss.item()

    epoch_test_loss = test_loss / (num_test//b) #test_loader.
    writer.add_scalar('Epoch_test_loss', epoch_test_loss, epoch)
    print('\t\t>>> Test epoch {} finished: epoch_test_loss: {}'.format(epoch, epoch_test_loss))

    model.train()

    if best_value > epoch_test_loss or best_value is None or epoch == args['epochs']:
        print('\t\t>>> New best: saving model ...')
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(args['save_dir'], 'epoch_{}.ckpt'.format(epoch)))
        print('\t\t>>> Model saved')
        best_value = epoch_test_loss



#visualise
#test = np.reshape(test_dataset[0],(10,25))

lats = d.variables['lat']
longs = d.variables['lon']

# together plot
fig, axes = plt.subplots(2, 1)

axes[0].set_title("Input")
m = Basemap(projection='merc',
                llcrnrlon=-70,  # lower longitude
                llcrnrlat=25,  # lower latitude
                urcrnrlon=50,  # uppper longitude
                urcrnrlat=70,  # uppper latitude
                resolution='i')

longs, lats = np.meshgrid(longs, lats)
#print(longs,lats)
xi, yi = m(longs, lats)
#dataset = denormalize(dataset)
inp = np.reshape(denormalize(dataset[11]), (10, 25))
print(inp)
plt.figure()
cs = m.pcolor(xi, yi, inp, shading='nearest', cmap='coolwarm', vmin=np.min(denormalize(dataset)), vmax=np.max(denormalize(dataset)))  # , vmin=0, vmax=1

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

#for output
test = dataset[0:100]
#test = test.reshape(-1, 250)
test = torch.tensor(test)
test = test.to(device)
#print(test[11])
tout = model(test.unsqueeze(1))[0]

dataset = denormalize(dataset)
axes[1].set_title("Output")
m = Basemap(projection='merc',
                llcrnrlon=-70,  # lower longitude
                llcrnrlat=25,  # lower latitude
                urcrnrlon=50,  # uppper longitude
                urcrnrlat=70,  # uppper latitude
                resolution='i')
#longs, lats = np.meshgrid(longs, lats)
#xi, yi = m(longs, lats)
inp2 = np.reshape(denormalize(tout[11].cpu().detach().numpy()), (10, 25))
print(inp2)
#inp2= preds2[0].cpu().detach().numpy()

plt.figure()
cs = m.pcolor(xi, yi, inp2, shading='nearest', cmap='coolwarm', vmin=np.min(dataset), vmax=np.max(dataset))  # vmin=0, vmax=1

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

plt.show()

print("The Mape")
print(mape(inp,inp2))