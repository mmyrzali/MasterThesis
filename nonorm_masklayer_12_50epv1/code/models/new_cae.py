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
    def __init__(self, input_channels, use_masklayer, masklayer_code_sizes,latent=70,mode=None):
        super().__init__()
        self.use_masklayer = use_masklayer
        self.masklayer_code_sizes = masklayer_code_sizes
        self.mode = mode

        self.masklayer = MaskLayer()
        self.bottleneck_size = 50  # Put here the number of features in the tensor you give as input to MaskLayer

        # >> Your code goes here: Encoder and decoder declaration <<
        # Encoder block
        self.layer1 = nn.Conv2d(1, 2, kernel_size=(2, 2),stride=2)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.MaxPool2d(2, stride=2)
        self.layer4 = nn.Conv2d(2, 4, kernel_size=3)
        self.layer5 = nn.ReLU()
        self.layer6 = nn.MaxPool2d(2, stride=2)
        self.layer7 = nn.Conv2d(4, 8, kernel_size=2)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.MaxPool2d(2, stride=2)
        self.layer10 = nn.Conv2d(8, 16, kernel_size=2)
        self.layer11 = nn.ReLU()
        self.layer12 = nn.Flatten()
        self.layer13 = nn.Linear(192,latent)



        # Decoder block
        self.layer14 = nn.Sequential(nn.Linear(latent, 192), nn.Unflatten(1, (16, 2, 6)))
        #self.layer11 = nn.Unflatten((1, torch.Size([16, 2, 5])))
        self.layer15 = nn.ReLU()
        self.layer16 = nn.ConvTranspose2d(16, 8, kernel_size=2)
        self.layer17 = nn.Upsample(size=[6, 14], mode='bilinear')
        self.layer18 = nn.ReLU()
        self.layer19 = nn.ConvTranspose2d(8, 4, kernel_size=2)
        self.layer20 = nn.Upsample(size=[14, 30], mode='bilinear')
        self.layer21 = nn.ReLU()
        self.layer22 = nn.ConvTranspose2d(4, 2, kernel_size=3)
        self.layer23 = nn.Upsample(size=[32, 64], mode='bilinear')
        self.layer24 = nn.ReLU()
        self.layer25 = nn.ConvTranspose2d(2, 1, kernel_size=2, stride=2)

    def forward(self, x, use_masklayer=None, masklayer_code_sizes=None, mode=None):
        if use_masklayer is None:
            use_masklayer = self.use_masklayer
            masklayer_code_sizes = self.masklayer_code_sizes

        if mode is None:
            x1 = self.layer1(x)
            #print("Output Layer 1: {}".format(x1.shape))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            # print(x3.shape)
            x4 = self.layer4(x3)
            # print(x4.shape)
            x5 = self.layer5(x4)
            x6 = self.layer6(x5)
            #print(x6.shape)
            x7 = self.layer7(x6)
            #print(x7.shape)
            x8 = self.layer8(x7)
            x9 = self.layer9(x8)
            #print(x9.shape)
            x10= self.layer10(x9)
            #print(x10.shape)
            x11 = self.layer11(x10)
            x12 = self.layer12(x11)

            encoded = self.layer13(x12)


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
        x14 = self.layer14(encoded) #,indices
        # print(x10.shape)
        x15 = self.layer15(x14)
        # print(x11.shape)
        x16 = self.layer16(x15)
        # print(x9.shape)
        x17 = self.layer17(x16)
        # print(x10.shape)
        x18 = self.layer18(x17)
        x19 = self.layer19(x18)
        x20 = self.layer20(x19)
        x21= self.layer21(x20)
        x22 = self.layer22(x21)
        x23 = self.layer23(x22)
        x24 = self.layer24(x23)
        decoded = self.layer25(x24)
        # print("decoded "+str(decoded.shape))


        return decoded, encoded
