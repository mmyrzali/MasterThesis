'''
MaskLayer your_net.py
Written by Remco Royen
'''

import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F


class MaskLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, code_sizes, bottleneck_size):
        code_size = np.random.choice(code_sizes)
        mask = torch.cat(
            [torch.ones([code_size], dtype=torch.float32, device=input.device),
             torch.zeros([bottleneck_size - code_size], dtype=torch.float32, device=input.device)], 0)
        mask = mask.unsqueeze(0).repeat([input.shape[0], 1])
        output = input * mask
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        mask = torch.nan_to_num(mask, nan=0)
        grad_output *= mask
        return grad_output, None, None


class YourNet(nn.Module):
    def __init__(self, input_channels, use_masklayer, masklayer_code_sizes,latent=30,mode=None):
        super().__init__()
        self.use_masklayer = use_masklayer
        self.masklayer_code_sizes = masklayer_code_sizes
        self.mode = mode

        self.masklayer = MaskLayer()
        self.bottleneck_size = 50  # Put here the number of features in the tensor you give as input to MaskLayer

        # >> Your code goes here: Encoder and decoder declaration <<
        # Encoder block
        self.layer1 = nn.Conv2d(1, 8, kernel_size=(2, 2), padding='same')
        self.layer2 = nn.ReLU()
        self.layer3 = nn.MaxPool2d(2, stride=2)
        self.layer4 = nn.Conv2d(8, 16, kernel_size=2)
        self.layer5 = nn.ReLU()
        self.layer6 = nn.MaxPool2d(2, stride=2)
        self.layer7 = nn.ReLU()
        self.layer8 = nn.Flatten()
        self.layer9 = nn.Linear(160,latent)



        # Decoder block
        self.layer10 = nn.Sequential(nn.Linear(latent, 160), nn.Unflatten(1, (16, 2, 5)))
        #self.layer11 = nn.Unflatten((1, torch.Size([16, 2, 5])))
        self.layer11 = nn.ReLU()
        self.layer12 = nn.Upsample(size=[4, 11], mode='bilinear')
        self.layer13 = nn.ReLU()
        self.layer14 = nn.ConvTranspose2d(16, 8, kernel_size=2)
        self.layer15 = nn.Upsample(size=[10, 25], mode='bilinear')
        self.layer16 = nn.ReLU()
        self.layer17 = nn.ConvTranspose2d(8, 1, kernel_size=1)

    def forward(self, x, use_masklayer=None, masklayer_code_sizes=None, mode=None):
        if use_masklayer is None:
            use_masklayer = self.use_masklayer
            masklayer_code_sizes = self.masklayer_code_sizes

        if mode is None:
            x1 = self.layer1(x)
            # print("Output Layer 1: {}".format(x1.shape))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            # print(x3.shape)
            x4 = self.layer4(x3)
            # print(x4.shape)
            x5 = self.layer5(x4)
            x6 = self.layer6(x5)
            x7 = self.layer7(x6)
            x8 = self.layer8(x7)

            encoded = self.layer9(x8)


        #, indices
        # print("encoded"+ str(encoded.shape))
        # print("Ind1"+str(indices1.shape))
        # print("Ind2"+str(indices2.shape))


        # Give a flattened tensor x: BxF
        if use_masklayer:
            encoded = MaskLayer.apply(encoded, masklayer_code_sizes, self.bottleneck_size)
            #print(encoded)
        # Returns a tensor of the same shape as input: x: BxF
        if mode:
            encoded = x


        # >> Your Decoder code goes here <<
        x10 = self.layer10(encoded) #,indices
        # print(x10.shape)
        x11 = self.layer11(x10)
        # print(x11.shape)
        x12 = self.layer12(x11)
        # print(x9.shape)
        x13 = self.layer13(x12)
        # print(x10.shape)
        x14 = self.layer14(x13)
        x15 = self.layer15(x14)
        x16 = self.layer16(x15)
        #x17 = self.layer17(x16)
        decoded = self.layer17(x16)
        # print("decoded "+str(decoded.shape))


        return decoded, encoded
