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
    def __init__(self, input_channels, use_masklayer, masklayer_code_sizes,latent=60,mode=None):
        super().__init__()
        self.use_masklayer = use_masklayer
        self.masklayer_code_sizes = masklayer_code_sizes
        self.mode = mode

        self.masklayer = MaskLayer()
        self.bottleneck_size = 50  # Put here the number of features in the tensor you give as input to MaskLayer

        # >> Your code goes here: Encoder and decoder declaration <<
        # Encoder block
        self.layer1 = nn.Conv2d(1, 8, kernel_size=(5, 5))
        self.layer2 = nn.ReLU()
        self.layer3 = nn.MaxPool2d(2, stride=2)
        self.layer4 = nn.Conv2d(8, 16, kernel_size=(5,5))
        self.layer5 = nn.ReLU()
        self.layer6 = nn.MaxPool2d(2, stride=2)
        self.layer7 = nn.Flatten()
        self.layer8 = nn.Linear(1040,500)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.Linear(500, latent)



        # Decoder block
        self.layer11 = nn.Linear(latent,500)
        self.layer12 = nn.ReLU()
        self.layer13 = nn.Sequential(nn.Linear(500, 1040), nn.Unflatten(1, (16, 5, 13)))
        #self.layer11 = nn.Unflatten((1, torch.Size([16, 2, 5])))
        self.layer14 = nn.Upsample(size=[10, 26], mode='bilinear')
        self.layer15 = nn.ReLU()
        self.layer16 = nn.ConvTranspose2d(16, 8, kernel_size=5)
        self.layer17 = nn.Upsample(size=[28, 60], mode='bilinear')
        self.layer18 = nn.ReLU()
        self.layer19 = nn.ConvTranspose2d(8, 1, kernel_size=5)

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
            x9 = self.layer9(x8)

            encoded = self.layer10(x9)


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
        x11 = self.layer11(encoded) #,indices
        # print(x10.shape)
        x12 = self.layer12(x11)
        # print(x11.shape)
        x13 = self.layer13(x12)
        # print(x9.shape)
        x14 = self.layer14(x13)
        # print(x10.shape)
        x15 = self.layer15(x14)
        x16 = self.layer16(x15)
        x17 = self.layer17(x16)
        x18 = self.layer18(x17)
        decoded = self.layer19(x18)
        # print("decoded "+str(decoded.shape))


        return decoded, encoded
