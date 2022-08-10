import torch.nn as nn
import torch
import torch.nn.functional as F

affine_par = True
from ConvGRU import ConvGRUCell
import cv2
import numpy as np
import torch
import torch.nn.init

model_urls = {'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
              'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}


# 两个预训练模型

class SNetModel(nn.Module):  # 定义S
    def __init__(self, middir):
        super(SNetModel, self).__init__()

        self.extra_convs = nn.Conv2d(1024, 8, 1)  # convs 输入通道数1024，输出通道数8，卷积核大小1*1
        all_channel = 28
        self.extra_conv_fusion = nn.Conv2d(all_channel * 2, all_channel, kernel_size=1,
                                           bias=True)  # conv_fusion:input:56-->output:28,核大小1，有偏置
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)  # LSTM模块 28-->28

        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)  # conv2d:28-->1,1*1
        self.extra_gate_s = nn.Sigmoid()  # sigmoid()激活函数

        self.extra_projf = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2,
                                     kernel_size=1)  # conv：28-->14
        self.extra_projg = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel, out_channels=all_channel, kernel_size=1)  # conv:28-->28

        self.middir = middir

        for m in self.modules():  # 权重初始化
            if isinstance(m, nn.Conv2d):  # 如果是二维卷积
                m.weight.data.normal_(0, 0.01)  # 正态分布
            elif isinstance(m, nn.BatchNorm2d):  # 是归一
                m.weight.data.fill_(1)  # 权重w全1
                m.bias.data.zero_()  # 偏置b为0

        net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        net = list(net.children())               # 列出 net 的外面7组
        self.features = nn.Sequential(*net[:7])  # 取前7组

    def forward(self, idx, img_name1, input1, epoch=1, label=None, index=None):

        x1 = self.features(input1)  # 1,512,32,32  resnext的网络结构
        x1 = self.extra_convs(x1)  # 1,28,32,32
        x2 = self.extra_conv_fusion(torch.cat((x1, self.self_attention(x1)), 1))
        x2 = self.extra_ConvGRU(x2, x1)

        self.map_1 = x1.clone()  # 1,28,32,32
        x1ss = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)  # 1,28,1,1
        x1ss = x1ss.view(-1, 28)  # 1,28

        self.map_2 = x2.clone()  # 1,28,32,32
        x2ss = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)  # 1,28,1,1
        x2ss = x2ss.view(-1, 28)  # 1,28

        return x1ss, x2ss, self.map_1, self.map_2

    def self_attention(self, x):
        m_batchsize, C, width, height = x.size()  # 8,28,32,32

        f = self.extra_projf(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H) # 8,14,1024
        g = self.extra_projg(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H) # 8,14,1024
        h = self.extra_projh(x).view(m_batchsize, -1, width * height)  # B * C * (W * H)      # 8,28,1024

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = F.softmax(attention, dim=1)  # 8,1024,1024

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)  # B * C * W * H
        self_mask = self.extra_gate(self_attetion)  # [1, 1, 32, 32]
        self_mask = self.extra_gate_s(self_mask)
        out = self_mask * x
        return out

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():
            # extra_projf.weight
            # extra_projf.bias
            # extra_projg.weight
            # extra_projg.bias
            # extra_projh.weight
            # extra_projh.bias
            # features.0.weight
            # features.1.weight
            # features.1.bias
            if 'extra' in name:                 # 作者自定义的结构
                if 'weight' in name:            # 权重和偏置分开存储
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:                               # resnext 中的结构
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups                           # 分成不同的参数组方便使用不同的lr

