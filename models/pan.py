import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
# import archs.PANutil as arch_util
import numpy as np
from torch.nn import init
#下面模块是从SRCNN网络添加的
from torch import nn

# from basicsr.utils.registry import ARCH_REGISTRY
from models import register
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
import math
import pdb
import time
import numpy as np
from math import sqrt
# import argparse
from argparse import Namespace

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


#Channel Attention (CA) Layer
class Ours_HCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Ours_HCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.contrast = stdv_channels
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, _, _ = x.size()
        channel_avg = self.avg_pool(x)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + 1e-5
        channel_std = channel_var.sqrt().unsqueeze(3)

        y = self.conv_du(channel_avg + channel_std)

        return x * y

# class EMA(nn.Module):
#     def __init__(self, channel, c2=None, factor=32):
#         super(EMA, self).__init__()
#         self.groups = factor
#         assert channel // self.groups > 0
#         self.softmax = nn.Softmax(-1)
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.gn = nn.GroupNorm(channel // self.groups, channel // self.groups)
#         self.conv1x1 = nn.Conv2d(channel // self.groups, channel // self.groups, kernel_size=1, stride=1, padding=0)
#         self.conv3x3 = nn.Conv2d(channel // self.groups, channel // self.groups, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         b, c, h, w = x.size()
#         group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
#         x_h = self.pool_h(group_x)
#         x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
#         hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
#         x_h, x_w = torch.split(hw, [h, w], dim=2)
#         x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
#         x2 = self.conv3x3(group_x)
#         x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
#         x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
#         weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
#         return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# class ChannelAttention(nn.Module):
#     def __init__(self,channel,reduction=16):
#         super().__init__()
#         self.maxpool=nn.AdaptiveMaxPool2d(1)
#         self.avgpool=nn.AdaptiveAvgPool2d(1)
#         self.se=nn.Sequential(
#             nn.Conv2d(channel,channel//reduction,1,bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel//reduction,channel,1,bias=False)
#         )
#         self.sigmoid=nn.Sigmoid()
    
#     def forward(self, x) :
#         max_result=self.maxpool(x)
#         avg_result=self.avgpool(x)
#         max_out=self.se(max_result)
#         avg_out=self.se(avg_result)
#         output=self.sigmoid(max_out+avg_out)
#         return output

# class SpatialAttention(nn.Module):
#     def __init__(self,kernel_size=7):
#         super().__init__()
#         self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
#         self.sigmoid=nn.Sigmoid()
    
#     def forward(self, x) :
#         max_result,_=torch.max(x,dim=1,keepdim=True)
#         avg_result=torch.mean(x,dim=1,keepdim=True)
#         result=torch.cat([max_result,avg_result],1)
#         output=self.conv(result)
#         output=self.sigmoid(output)
#         return output

# class CBAMBlock(nn.Module):

#     def __init__(self, channel=512,reduction=16,kernel_size=49):
#         super().__init__()
#         self.ca=ChannelAttention(channel=channel,reduction=reduction)
#         self.sa=SpatialAttention(kernel_size=kernel_size)


#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         residual=x
#         out=x*self.ca(x)
#         out=out*self.sa(out)
#         return out+residual

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0],-1)

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16,num_layers=3):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        gate_channels=[channel]
        gate_channels+=[channel//reduction]*num_layers
        gate_channels+=[channel]


        self.ca=nn.Sequential()
        self.ca.add_module('flatten',Flatten())
        for i in range(len(gate_channels)-2):
            self.ca.add_module('fc%d'%i,nn.Linear(gate_channels[i],gate_channels[i+1]))
            self.ca.add_module('bn%d'%i,nn.BatchNorm1d(gate_channels[i+1]))
            self.ca.add_module('relu%d'%i,nn.ReLU())
        self.ca.add_module('last_fc',nn.Linear(gate_channels[-2],gate_channels[-1]))
        

    def forward(self, x) :
        res=self.avgpool(x)
        res=self.ca(res)
        res=res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res

class SpatialAttention(nn.Module):
    def __init__(self,channel,reduction=16,num_layers=3,dia_val=2):
        super().__init__()
        self.sa=nn.Sequential()
        self.sa.add_module('conv_reduce1',nn.Conv2d(kernel_size=1,in_channels=channel,out_channels=channel//reduction))
        self.sa.add_module('bn_reduce1',nn.BatchNorm2d(channel//reduction))
        self.sa.add_module('relu_reduce1',nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d'%i,nn.Conv2d(kernel_size=3,in_channels=channel//reduction,out_channels=channel//reduction,padding=1,dilation=dia_val))
            self.sa.add_module('bn_%d'%i,nn.BatchNorm2d(channel//reduction))
            self.sa.add_module('relu_%d'%i,nn.ReLU())
        # self.sa.add_module('last_conv',nn.Conv2d(channel//reduction,1,kernel_size=1)) //source code
        self.sa.add_module('last_conv',nn.Conv2d(channel//reduction,1,kernel_size=1,padding=3))

    def forward(self, x) :
        res=self.sa(x)
        res=res.expand_as(x)
        return res

class BAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,dia_val=2):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(channel=channel,reduction=reduction,dia_val=dia_val)
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out=self.sa(x)
        ca_out=self.ca(x)
        weight=self.sigmoid(sa_out+ca_out)
        out=(1+weight)*x
        return out

######################多尺度金字塔蒸馏特征提取模块#####################

class Multi_DENSE_A(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(Multi_DENSE_A, self).__init__()

        self.bam = BAMBlock(in_channels)

        self.c2_1 = conv_layer(48, 48, 3)
        self.c3_1 = conv_layer(32, 32, 3)
        self.c4_1 = conv_layer(16, 16, 3)
        self.act = activation('lrelu', neg_slope=0.2)
        self.conv1x1 = conv_layer(in_channels, in_channels, 1, bias=True)
        
        #注释这六行-做只修改模块B的消融实验
        self.hca_a2 = Ours_HCALayer(48)
        self.hca_a3 = Ours_HCALayer(32)
        self.hca_a4 = Ours_HCALayer(16)
        self.conv1x1_a2 = conv_layer(48, 48, 1, bias=True)
        self.conv1x1_a3 = conv_layer(32, 32, 1, bias=True)
        self.conv1x1_a4 = conv_layer(16, 16, 1, bias=True)
        
 

    def forward(self, input):
        input = self.bam(input)

        distilled_c1, distilled_c2_0 = torch.split(input, (16, 48), dim=1)
        distilled_c2 = self.act(self.c2_1(distilled_c2_0) + distilled_c2_0)
      

        distilled_c2 = self.conv1x1_a2(distilled_c2) #注释这1行-做只修改模块B的消融实验
        distilled_c2 = self.hca_a2(distilled_c2) #注释这1行-做只修改模块B的消融实验
        # distilled_c2 = self.conv_du2(distilled_c2)
        distilled_c2 = distilled_c2 + distilled_c2_0 #注释这1行-做只修改模块B的消融实验

        distilled_c2, distilled_c3_0 = torch.split(distilled_c2, (16, 32), dim=1)
        distilled_c3 = self.act(self.c3_1(distilled_c3_0) + distilled_c3_0)


        distilled_c3 = self.conv1x1_a3(distilled_c3) #注释这1行-做只修改模块B的消融实验
        distilled_c3 = self.hca_a3(distilled_c3)  #注释这1行-做只修改模块B的消融实验
        # distilled_c3 = self.conv_du3(distilled_c3)
        distilled_c3 = distilled_c3 + distilled_c3_0  #注释这1行-做只修改模块B的消融实验

        distilled_c3, distilled_c4_0 = torch.split(distilled_c3, (16, 16), dim=1)
        distilled_c4 = self.act(self.c4_1(distilled_c4_0) + distilled_c4_0)

        distilled_c4 = self.conv1x1_a4(distilled_c4)  #注释这1行-做只修改模块B的消融实验
        distilled_c4 = self.hca_a4(distilled_c4)   #注释这1行-做只修改模块B的消融实验
        # distilled_c4 = self.conv_du4(distilled_c4)
        distilled_c4 = distilled_c4 + distilled_c4_0  #注释这1行-做只修改模块B的消融实验

        out_0 = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)
        out = self.conv1x1(out_0)

        return out


class Multi_DENSE_B(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(Multi_DENSE_B, self).__init__()

        self.c2_1 = conv_layer(16, 16, 3)
        self.c3_1 = conv_layer(32, 32, 3)
        self.c4_1 = conv_layer(48, 48, 3)
        self.act = activation('lrelu', neg_slope=0.2)
        self.conv1x1 = conv_layer(in_channels, in_channels, 1, bias=True)

        self.hca_b2 = Ours_HCALayer(16)
        self.hca_b3 = Ours_HCALayer(32)
        self.hca_b4 = Ours_HCALayer(48)

        self.conv1x1_b2 = conv_layer(16, 16, 1, bias=True) #这三行先去掉，做修改A模块的消融实验
        self.conv1x1_b3 = conv_layer(32, 32, 1, bias=True)
        self.conv1x1_b4 = conv_layer(48, 48, 1, bias=True)

       

    def forward(self, input):
        distilled_c1_0, distilled_c2, distilled_c3, distilled_c4 = torch.split(input, (16, 16, 16, 16), dim=1)
        distilled_c1 = self.act(self.c2_1(distilled_c1_0) + distilled_c1_0)

        distilled_c1 = self.conv1x1_b2(distilled_c1)  #这一行先去掉，做修改A模块的消融实验


        distilled_c2_0 = torch.cat([distilled_c1, distilled_c2], dim=1)
        distilled_c2 = self.act(self.c3_1(distilled_c2_0) + distilled_c2_0)
    

        distilled_c2 = self.conv1x1_b3(distilled_c2)  #这一行先去掉，做修改A模块的消融实验


        distilled_c3_0 = torch.cat([distilled_c2, distilled_c3], dim=1)
        distilled_c3 = self.act(self.c4_1(distilled_c3_0) + distilled_c3_0)

        distilled_c3 = self.conv1x1_b4(distilled_c3)  #这一行先去掉，做修改A模块的消融实验


        out = torch.cat([distilled_c3, distilled_c4], dim=1)

        out = self.conv1x1(out)

        return out


#####################第三章模型########################
class IMDModule_OURS(nn.Module):
    def __init__(self, in_channels):
        super(IMDModule_OURS, self).__init__()

        self.conv1 = Multi_DENSE_A(in_channels)
        self.conv2 = Multi_DENSE_B(in_channels)
        self.act = activation('lrelu', neg_slope=0.2)
        # self.hca = Ours_HCALayer(in_channels)  # 整个网络都不用注意力
        self.bam = BAMBlock(in_channels)

        #self.nam = NAM(in_channels)

        # self.conv1x1 = conv_layer(in_channels, in_channels, 1, bias=True)

    def forward(self, input):



        distilled_c1 = self.conv1(input)
        # out = self.conv1x1(distilled_c1)
        out = self.act(distilled_c1)
        #out = self.act(distilled_c1)
        out = self.conv2(out)
        # out = self.hca(out)  # 整个网络都不用注意力
        out = self.bam(out)

        # out_11 = self.hca(distilled_c1)
        # out_11 =  self.conv1x1(out_11)

        out = out + input # +out_11
        return out

# @ARCH_REGISTRY.register()
class PAN(nn.Module):

    def __init__(self, args, conv=default_conv):
        super(PAN, self).__init__()
        self.args = args
        num_in_ch = args.num_in_ch
        num_out_ch = args.num_out_ch
        nf = args.nf
        unf = args.unf
        nb = args.nb
        upscale = args.scale[0]
        kernel_size = 3

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, nf, kernel_size)]

        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # SCPA
        #######第三章模型#############
        # SCPA_block_f = functools.partial(IMDModule_OURS, in_channels=nf) #nb = 12
        #######第四章模型#############
        SCPA_block_f = functools.partial(IMDModule_OURS, in_channels=nf)  # nb = 24
        ############################
        self.scale = upscale

        ### first convolution
        self.conv_first = nn.Conv2d(num_in_ch, nf, 3, 1, 1, bias=True)

        ### main blocks
        # self.SCPA_trunk = arch_util.make_layer(SCPA_block_f, nb)
        self.SCPA_trunk_1 = make_layer(SCPA_block_f, nb)
        # self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.nam = NAM(nf)
        self.trunk_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # self.trunk_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        # self.att1 = HOALayer(unf,1)

        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        # if self.scale == 4:
        #     self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        #     # self.att2 = HOALayer(unf,2)
        #     self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        # elif self.scale == 8:
        #     self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        #     # self.att2 = HOALayer(unf,2)
        #     self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        #     self.upconv3 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        #     # self.att3 = HOALayer(unf,3)
        #     self.HRconv3 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if args.no_upsampling:
            self.out_dim = nf
        else:
            self.out_dim = args.n_colors
            if self.scale == 4:
                self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
                # self.att2 = HOALayer(unf,2)
                self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            elif self.scale == 8:
                self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
                # self.att2 = HOALayer(unf,2)
                self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
                self.upconv3 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
                # self.att3 = HOALayer(unf,3)
                self.HRconv3 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
           

        self.conv_last = nn.Conv2d(unf, num_out_ch, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        fea = self.conv_first(x)

        #nam_1 = self.nam(self.SCPA_trunk_1(fea))
        # nam_2 = fea + nam_1 
        # trunk_1 = self.trunk_conv1(nam_2)
        trunk_1 = self.trunk_conv1(self.SCPA_trunk_1(fea))
        fea = fea + trunk_1


        if self.args.no_upsampling:
            x = fea
        else:   
            if self.scale == 2 or self.scale == 3:
                fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
                # fea = self.lrelu(self.att1(fea))
                fea = self.lrelu(self.HRconv1(fea))
            elif self.scale == 4:
                fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            # fea = self.lrelu(self.att1(fea))
                fea = self.lrelu(self.HRconv1(fea))
                fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            # fea = self.lrelu(self.att2(fea))
                fea = self.lrelu(self.HRconv2(fea))
            elif self.scale == 8:
                fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            # fea = self.lrelu(self.att1(fea))
                fea = self.lrelu(self.HRconv1(fea))
                fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            # fea = self.lrelu(self.att2(fea))
                fea = self.lrelu(self.HRconv2(fea))
                fea = self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest'))
            # fea = self.lrelu(self.att3(fea))
                fea = self.lrelu(self.HRconv3(fea))
            high = self.conv_last(fea)
            ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
            x = high + ILR
        # out = high

        return x

@register('pan')
def make_pan(num_in_ch=3, num_out_ch=3, nf=64, unf=24, nb=24, upscale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.num_in_ch = num_in_ch
    args.num_out_ch = num_out_ch
    args.nf = nf
    args.unf = unf
    args.nb = nb
    args.scale = [upscale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range
    args.res_scale = 1
    args.n_colors = 3
    return PAN(args)