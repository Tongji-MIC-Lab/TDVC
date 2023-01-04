from __future__ import absolute_import

import torch
import torch.nn as nn
import math

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn import functional as F


def inflate_conv(conv2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, 3, 3)
    padding = (time_padding, 1, 1)
    stride = (time_stride, 1, 1)
    dilation = (time_dilation, 1, 1)
    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = nn.Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    """
    Args:
        time_dim: final time dimension of the features
    """
    linear3d = nn.Linear(linear2d.in_features * time_dim,
                         linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = nn.Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
    padding = (time_padding, pool2d.padding, pool2d.padding)
    if time_stride is None:
        time_stride = time_dim
    stride = (time_stride, pool2d.stride, pool2d.stride)
    if isinstance(pool2d, nn.MaxPool2d):
        dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
        pool3d = nn.MaxPool3d(
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
            ceil_mode=pool2d.ceil_mode)
    elif isinstance(pool2d, nn.AvgPool2d):
        pool3d = nn.AvgPool3d(kernel_dim, stride=stride)
    else:
        raise ValueError(
            '{} is not among known pooling classes'.format(type(pool2d)))
    return pool3d


class APP3DC(nn.Module):
    def __init__(self):
        super(APP3DC, self).__init__()
        # spatial conv3d kernel
        self.spatial_conv3d = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1))
        # temporal conv3d kernel
        self.temporal_conv3d = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=(1, 1, 1), bias=False)

    def forward(self, x):
        out = self.spatial_conv3d(x)
        residual = self.temporal_conv3d(out)
        out = out + residual
        return out


class Bottleneck3D(nn.Module):
    def __init__(self, inc=64):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inc, inc, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.spatial_conv3d = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1))
        self.temporal_conv3d = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=(1, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(inc, inc, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)

        out = self.spatial_conv3d(out)
        tout = self.temporal_conv3d(out)
        out = out + tout

        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        out = self.relu(out)

        return out


class SELayerFc(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
        init_cfg (dict or list[dict], optional): Initialization cfg dict.
            Default: None
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 init_cfg=None):
        super(SELayer, self).__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class Res21D_Block(nn.Module):
    def __init__(self, in_channel, out_channel, spatial_stride=1, temporal_stride=1):
        super(Res21D_Block, self).__init__()
        self.MidChannel1 = int((27 * in_channel * out_channel) / (9 * in_channel + 3 * out_channel))
        self.MidChannel2 = int((27 * out_channel * out_channel) / (12 * out_channel))
        self.conv1_2D = nn.Conv3d(in_channel, self.MidChannel1, kernel_size=(1, 3, 3),
                                  stride=(1, spatial_stride, spatial_stride),
                                  padding=(0, 1, 1))
        self.conv1_1D = nn.Conv3d(self.MidChannel1, out_channel, kernel_size=(3, 1, 1), stride=(temporal_stride, 1, 1),
                                  padding=(1, 0, 0))
        self.conv2_2D = nn.Conv3d(out_channel, self.MidChannel2, kernel_size=(1, 3, 3), stride=1,
                                  padding=(0, 1, 1))
        self.conv2_1D = nn.Conv3d(self.MidChannel2, out_channel, kernel_size=(3, 1, 1), stride=1,
                                  padding=(1, 0, 0))

        self.relu = nn.ReLU()

    def forward(self, x):
        x_branch = self.conv1_2D(x)
        x_branch = self.relu(x_branch)
        x_branch = self.conv1_1D(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2_2D(x_branch)
        x_branch = self.relu(x_branch)
        x_branch = self.conv2_1D(x_branch)

        return self.relu(x_branch + x)
