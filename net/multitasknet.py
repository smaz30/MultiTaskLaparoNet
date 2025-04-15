from __future__ import absolute_import, division, print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext2 import ConvNeXtV2
from collections import OrderedDict

def convnextv2_pico(**kwargs):
    dims = [64, 128, 256, 512]
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims= dims, **kwargs)
    return model, dims

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    dims = [ 80, 160, 320, 640]
    return model, dims

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    dims=[ 96, 192, 384, 768]
    return model, dims

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    dims = [128, 256, 512, 1024]
    return model, dims

class ConvBlockNorm(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, norm = 'batch'):
        super(ConvBlockNorm, self).__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def upsample(x, mode = 'nearest', scale = 2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale, mode=mode, align_corners= False)

class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class ConvNormBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, normalization_mode = 'batch'):
        super(ConvNormBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size=kernel_size, padding=1)
        if normalization_mode == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif normalization_mode == 'layer':
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()
        self.activation = nn.GELU()

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
class DecoderBlock(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, upsampling_mode = 'bilinear', normalization_mode = 'layer', upsample_scale = 2):
        super(DecoderBlock, self).__init__()
        self.upsamplimg_mode = upsampling_mode
        self.normalization_mode = normalization_mode
        self.upsample_scale = upsample_scale
        if in_channels_2 is not None:
            self.block_1 = ConvNormBlock(in_channels_1 + in_channels_2, in_channels_1, normalization_mode=self.normalization_mode)
            self.block_2 = ConvNormBlock(in_channels_1, out_channels, normalization_mode=self.normalization_mode)
        else:
            self.block_1 = ConvNormBlock(in_channels_1 , in_channels_1, normalization_mode=self.normalization_mode)
            self.block_2 = ConvNormBlock(in_channels_1 , out_channels, normalization_mode=self.normalization_mode)
    def forward(self,x1,x2):
        if x2 is not None:
            x1 = upsample(x1, mode=self.upsamplimg_mode,scale=self.upsample_scale)
            x3 = torch.cat([x1,x2], dim= 1)
            x3 = self.block_1(x3)
            x3 = self.block_2(x3)
            return x3
        else:
            x3 = upsample(x1, mode=self.upsamplimg_mode, scale= self.upsample_scale)
            x3 = self.block_1(x3)
            x3 = self.block_2(x3)
            return x3
    

class MultitaskDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_output_channels=1, ):
        super(MultitaskDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.upsample_mode = 'bilinear'

        self.encoder_model = 'convnext'
        self.num_ch_enc = num_ch_enc[::-1]
        self.number_of_skips = len(num_ch_enc) -1
        # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_dec = np.array([256,128,64,32,16])

        # decoder
        self.convs_1 = nn.ModuleList()
        self.convs_2 = nn.ModuleList()
        for i in range(self.number_of_skips):
            num_ch_in = self.num_ch_enc[i] if i == 0 else self.num_ch_dec[i-1]
            num_ch_out = self.num_ch_dec[i]
            num_ch_skip = self.num_ch_enc[i+1]
            self.convs_1.append(ConvBlockNorm(num_ch_in, num_ch_out,norm=None))
            self.convs_2.append(ConvBlockNorm(num_ch_out+num_ch_skip, num_ch_out))
        if self.encoder_model == 'resnet':
            num_ch_out = self.num_ch_dec[i]
            num_ch_skip = self.num_ch_enc[i+1]
            self.convs_1.append(ConvBlockNorm(num_ch_in, num_ch_out, norm=None))
            self.convs_2.append(ConvBlockNorm(num_ch_out+num_ch_skip, num_ch_out))
        elif self.encoder_model == 'convnext':
            num_ch_in = self.num_ch_dec[-3]
            int_ch = self.num_ch_dec[-2]
            num_ch_out = self.num_ch_dec[-1]
            
            self.convs_1.append(nn.ModuleList([ConvBlockNorm(num_ch_in, int_ch, norm=None), ConvBlockNorm(num_ch_in, int_ch, norm=None)]))
            self.convs_2.append(nn.ModuleList([ConvBlockNorm(int_ch, num_ch_out), ConvBlockNorm(int_ch, num_ch_out)]))

        

        self.conv_out = Conv3x3(self.num_ch_dec[-1], self.num_output_channels)
        self.conv_out_segmentation = Conv3x3(self.num_ch_dec[-1], self.num_output_channels)
        self.sigmoid = nn.Sigmoid()
        self.activation_segmentation = nn.Identity()

    def forward(self, input_features):
        

        # decoder
        x = input_features[-1]
        x = self.convs_1[0](x)
        x = upsample(x,mode=self.upsample_mode)
        x = torch.cat([x, input_features[-2]], dim=1)
        x = self.convs_2[0](x)

        x = self.convs_1[1](x)
        x = upsample(x,mode=self.upsample_mode)
        x = torch.cat([x, input_features[-3]], dim=1)
        x = self.convs_2[1](x)

        x = self.convs_1[2](x)
        x = upsample(x,mode=self.upsample_mode)
        x = torch.cat([x, input_features[-4]], dim=1)
        x_head = self.convs_2[2](x)
 
        x = self.convs_1[3][0](x_head)
        x = upsample(x,mode=self.upsample_mode)
        x = self.convs_2[3][0](x)
        x = upsample(x,mode=self.upsample_mode)

        y = self.convs_1[3][1](x_head)
        y = upsample(y,mode=self.upsample_mode)
        y = self.convs_2[3][1](y)
        y = upsample(y,mode=self.upsample_mode)


        out = self.alpha*(self.sigmoid(self.conv_out(x))) + self.beta

        out_segmentation = self.conv_out_segmentation(y)
        
        return out, out_segmentation
    
class MultitaskNet(nn.Module):

    def __init__(self, convnext_size = 'pico'):
        super(MultitaskNet, self).__init__()
        if convnext_size == 'pico':
            self.encoder,dims = convnextv2_pico()
        elif convnext_size == 'nano':
            self.encoder,dims = convnextv2_nano()
        elif convnext_size == 'tiny':
            self.encoder,dims = convnextv2_tiny()
        elif convnext_size == 'base':
            self.encoder,dims = convnextv2_base()
        self.decoder = MultitaskDecoder(dims)
        

    def init_weights(self):
        pass

    def forward(self, x):
        features = self.encoder(x)
        # print('feat',features[0].shape, features[1].shape, features[2].shape)
        
        disp, segmentation= self.decoder(features)
        
        return disp, segmentation, None
if __name__ == "__main__":

    dummy_in = torch.randn((4,3,256,320)).cuda()
    disp_net = MultitaskNet(convnext_size='pico').cuda()
    disp, mask ,_ = disp_net(dummy_in)
    print(disp.shape, mask.shape)
