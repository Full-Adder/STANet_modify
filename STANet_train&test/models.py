import torch
import torch.nn as nn
from Soundmodel import SoundNet

class att_Model(nn.Module):
    def __init__(self):  # 128,128,512,29
        super(att_Model, self).__init__()
        Amodel= SoundNet()
        checkpoint = torch.load('vggsound_netvlad.pth.tar')
        Amodel.load_state_dict(checkpoint['model_state_dict'])
        Amodel = list(Amodel.audnet.children())
        self.layerA = nn.Sequential(*Amodel[:9])

        net = torch.hub.load('facebookresearch/WSL-Images','resnext101_32x8d_wsl')
        net = list(net.children())
        self.layerV = nn.Sequential(*net[:9])

        self.layerA_d = nn.Linear(8192, 128)
        self.layerV_d = nn.Linear(2048, 128)

        self.extra_bilinear = nn.Bilinear(128, 128, 128)

        self.fc = nn.Linear(128, 2)

    def forward(self, audio, video_inputs):
        layerA = self.layerA(audio.unsqueeze(1))                         # 1,1,257,48->1,64,65,12
        layerA_p = layerA.reshape(layerA.size(0), -1)
        layerA_d = self.layerA_d(layerA_p)

        layerV = self.layerV(video_inputs)
        layerV_p = layerV.reshape(layerV.size(0), -1)
        layerV_d = self.layerV_d(layerV_p)
        
        layerAV = self.extra_bilinear(layerA_d, layerV_d)
        fc = self.fc(layerAV) # [8, 2048, 1, 1]

        return torch.index_select((fc > 0).float(), 1, torch.tensor(1).cuda()).view(fc.size(0),1,1,1)
