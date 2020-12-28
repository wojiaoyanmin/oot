import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init,bias_init_with_prob, ConvModule

from ..builder import HEADS, build_loss,build_head

import torch
import numpy as np
import pdb


@HEADS.register_module
class MaskFeatHead(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 start_level,
                 end_level,
                 num_convs,
                 num_classes,
                 conv_cfg=None,
                 norm_cfg=None):
        super(MaskFeatHead, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_convs):
            chn = self.in_channels if i==0 else self.mid_channels
            self.convs_all_levels.append(
                ConvModule(
                    chn,
                    self.mid_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.conv_pred = ConvModule(
                self.mid_channels,
                self.num_classes,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=None)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    def forward(self, inputs):
        feature = inputs[0]
        for conv in self.convs_all_levels:
            feature = conv(feature)
        feature_pred = self.conv_pred(feature)
        return feature_pred
