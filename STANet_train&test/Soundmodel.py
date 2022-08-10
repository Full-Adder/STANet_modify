import torch
from torch import nn
import torch.nn.functional as F
import resnet

class SoundNet(nn.Module):

    def __init__(self):
        super(SoundNet, self).__init__()
        self.audnet = Resnet()

    def forward(self, audio):
        aud = self.audnet(audio)
        return aud


def Resnet():
    model = resnet.resnet18(num_classes=309, pool="vlad")
    return model 



