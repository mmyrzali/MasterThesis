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
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', help='C:/AEcode/configs/config1.json', default="./configs/config1.json")
args = parser.parse_args()
config = args.config_path

print('Initialization')
print('\t>>> Loading {}'.format(args.config_path))

args = json.load(open(args.config_path))
save_tr_code(args['save_dir'], config, 'train.py')

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
#data_all = normalize(data_all)
minx = np.min(data_all)
maxx = np.max(data_all)
dataset = data_all[0:44997]
test_dataset = data_all[44997:]

def denormalize(x):
    return x*(maxx-minx)+minx

print('Size of train data: '+str(len(dataset)))
print('Size of test data: '+str(len(test_dataset)))



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
#model = model.float()
for epoch in range(start_epoch, args['epochs']+1):
    print('\t>>> Epoch: %d' % epoch)
    #encods1 = np.empty((0, 50))
    # best_encoded1 = np.empty((0, 50))
    # best_encoded2 = np.empty((0, 50))
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

        preds1 = model(data.unsqueeze(1), use_masklayer=args['use_masklayer'], masklayer_code_sizes=args['masklayer_code_sizes'])[0]
        #latent = model(data.unsqueeze(1), use_masklayer=args['use_masklayer'], masklayer_code_sizes=args['masklayer_code_sizes'])[1]
        #latent = latent.cpu().detach().numpy()
        #print("Latent shape"+str(latent.shape))
        #encods1 = np.vstack((encods1, latent))

        loss = loss_function(preds1, data.unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar('Batch_tr_loss', loss.item()/b, i + num_samples//b*epoch)

        tr_loss += loss.item()

    #full_encods1 = encods1

    epoch_tr_loss = tr_loss /(num_samples//b) #loader.
    writer.add_scalar('Epoch_tr_loss', epoch_tr_loss, epoch)
    print('\t\t>>> Epoch {} finished: epoch_tr_loss: {}'.format(epoch, epoch_tr_loss))

    model.eval()
    test_loss = 0
    start = 0
    end=0
    #encods2 = np.empty((0, 50))
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
        #latent2 = model(data_t.unsqueeze(1), use_masklayer=args['test_use_masklayer'], masklayer_code_sizes=args['test_code_sizes'])[1]
        #latent2 = latent2.cpu().detach().numpy()
        #print("Latent2 size" + str(latent2.shape))
        #encods2 = np.vstack((encods2, latent2))
        loss = loss_function(preds2, data_t.unsqueeze(1))
        writer.add_scalar('Batch_test_loss', loss.item()/b, i + num_test//b*epoch)
        test_loss += loss.item()

    #full_encods2 = encods2
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
        #best_encoded1 = full_encods1
        #best_encoded2 = full_encods2

#all_encoded = np.vstack((best_encoded1,full_encods2))
#print("All embeddings size "+str(all_encoded.shape))


