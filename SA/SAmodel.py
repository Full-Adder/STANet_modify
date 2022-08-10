import torch.nn as nn
import torch
import torch.nn.functional as F
affine_par = True
from ConvGRU import ConvGRUCell
import cv2
import numpy as np
from Soundmodel import SoundNet

class SANetModel(nn.Module):
    def  __init__(self, middir):
        super(SANetModel, self).__init__()
        
        self.extra_audio_d = nn.Linear(8192, 2048)
        self.Aup = nn.Sequential(nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=0), nn.ReLU(True),
                                 nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=0), nn.ReLU(True),
                                 nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=1, padding=0)) 

        self.extra_video_d = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, padding=1), nn.ReLU(True),
                                           nn.Conv2d(2048, 2048, kernel_size=3, padding=1), nn.ReLU(True),
                                           nn.Conv2d(2048, 2048, kernel_size=3, padding=1), nn.ReLU(True),
                                           nn.Conv2d(2048, 28, 1))

        self.extra_convs = nn.Sequential(nn.Conv2d(2048, 28, 1), nn.Conv2d(28, 1, 1), nn.Sigmoid()) 
        all_channel = 28
        self.extra_conv_fusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=1, bias=True)
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)

        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.extra_gate_s = nn.Sigmoid()

        self.extra_projf = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projg = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel, out_channels=all_channel, kernel_size=1)

        self.middir = middir

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  

        Amodel = SoundNet()
        checkpoint = torch.load('vggsound_netvlad.pth.tar')
        Amodel.load_state_dict(checkpoint['model_state_dict'])
        Amodel = list(Amodel.audnet.children())
        self.audio_model = nn.Sequential(*Amodel[:9])  

        net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        net = list(net.children())
        self.features = nn.Sequential(*net[:8]) 
    		
    def forward(self, idx,  img_name1, input1, audio1, epoch=1, label=None, index=None):
        batch_num  = input1.size()[0]
        a1 = self.audio_model(audio1.unsqueeze(1)) # [13, 8192]
        a1 = self.extra_audio_d(a1).unsqueeze(2) # [16, 2048, 1] 
        Aup = self.Aup(a1.unsqueeze(2))# 1,2048,8,8

        x1 = self.features(input1) # 1,2048,8,8
        
        x1 = self.extra_video_d(x1)
        x2 = self.extra_conv_fusion(torch.cat((F.relu(x1+self.self_attention(x1)), F.relu(x1+x1*self.extra_convs(Aup))), 1))
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

        f = self.extra_projf(x).view(m_batchsize, -1, width * height)
        g = self.extra_projg(x).view(m_batchsize, -1, width * height)
        h = self.extra_projh(x).view(m_batchsize, -1, width * height)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = F.softmax(attention, dim=1)  # 8,1024,1024

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(
            m_batchsize, C, width, height)  # B * C * W * H
        self_mask = self.extra_gate(self_attetion)  # [1, 1, 32, 32]
        self_mask = self.extra_gate_s(self_mask)
        out = self_mask * x
        return out  # [1, 28, 32, 32]
 
    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():
            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups
