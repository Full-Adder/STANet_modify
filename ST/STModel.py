import torch.nn as nn
import torch
import torch.nn.functional as F
affine_par = True
from ConvGRU import ConvGRUCell

model_urls = {'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class STNetModel(nn.Module):
    def  __init__(self, middir):
        super(STNetModel, self).__init__()
        self.extra_convs = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(2048, 28, 1))
        all_channel = 28
        self.extra_conv_fusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=True)
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.extra_gate_s = nn.Sigmoid()
        self.extra_projf = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projg = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel, out_channels=all_channel, kernel_size=1)
        self.extra_refineST = nn.Sequential(
            nn.Conv3d(28, 28, (3, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(28, 28, (1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(28, 28, (1, 3, 3), padding=(0, 1, 1)))
        self.extra_s = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        net = list(net.children())
        self.features = nn.Sequential(*net[:8])
    		
    def forward(self, idx,  img_name1, img_name2, img_name3, input1, input2, input3, epoch=1, label=None, index=None):

        batch_num  = input1.size()[0]
 
        x1 = self.features(input1) # 1,512,8,8
        x1 = self.extra_convs(x1) # 1,28,8,8
        self.map_1     = torch.zeros(batch_num,28,8,8).cuda()
        self.map_all_1 = torch.zeros(batch_num,28,8,8).cuda()
        x1sss          = torch.zeros(batch_num, 28).cuda()
        self.map_1     = x1
        x1ss = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0) # 1,28,1,1
        x1ss = x1ss.view(-1, 28) # 1,28

        x2 = self.features(input2)
        x2 = self.extra_convs(x2) # 1,28,8,8
        self.map_2     = torch.zeros(batch_num,28,8,8).cuda()
        self.map_all_2 = torch.zeros(batch_num,28,8,8).cuda()
        x2sss          = torch.zeros(batch_num, 28).cuda()
        self.map_2     = x2
        x2ss = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        x2ss = x2ss.view(-1, 28)

        x3 = self.features(input3)
        x3 = self.extra_convs(x3) # 1,28,8,8
        self.map_3     = torch.zeros(batch_num,28,8,8).cuda()
        self.map_all_3 = torch.zeros(batch_num,28,8,8).cuda()
        x3sss          = torch.zeros(batch_num, 28).cuda()
        self.map_3     = x3
        x3ss = F.avg_pool2d(x3,kernel_size=(x3.size(2),x3.size(3)),padding=0)
        x3ss = x3ss.view(-1, 28) # 2,28
        
        incat1 = torch.cat((x2.unsqueeze(1), x1.unsqueeze(1), x3.unsqueeze(1)), 1).view(batch_num,28,3,8,8)
        incat2 = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), 1).view(batch_num,28,3,8,8)
        incat3 = torch.cat((x1.unsqueeze(1), x3.unsqueeze(1), x2.unsqueeze(1)), 1).view(batch_num,28,3,8,8)

        x11 = self.extra_conv_fusion(torch.cat((F.relu(x1+self.self_attention(x1)), F.relu(x1+x1*self.extra_s(self.extra_refineST(incat1).squeeze(2)))), 1))
        x11 = self.extra_ConvGRU(x11, x1)
        self.map_all_1 = x11
        x1sss = F.avg_pool2d(x11, kernel_size=(x11.size(2), x11.size(3)), padding=0)
        x1sss = x1sss.view(-1, 28) # 1,28

        x22 = self.extra_conv_fusion(torch.cat((F.relu(x2+self.self_attention(x2)), F.relu(x2+x2*self.extra_s(self.extra_refineST(incat2).squeeze(2)))), 1))
        x22 = self.extra_ConvGRU(x22, x2)
        self.map_all_2 = x22
        x2sss = F.avg_pool2d(x22, kernel_size=(x22.size(2), x22.size(3)), padding=0)         
        x2sss = x2sss.view(-1, 28) # 1,28

        x33 = self.extra_conv_fusion(torch.cat((F.relu(x3+self.self_attention(x3)), F.relu(x3+x3*self.extra_s(self.extra_refineST(incat3).squeeze(2)))), 1))
        x33 = self.extra_ConvGRU(x33, x3)
        self.map_all_3 = x33
        x3sss = F.avg_pool2d(x33, kernel_size=(x33.size(2), x33.size(3)), padding=0)         
        x3sss = x3sss.view(-1, 28) # 1,28
       
        return x1ss,x1sss, x2ss,x2sss, x3ss,x3sss, self.map_1, self.map_all_1

    def self_attention(self, x):
        m_batchsize, C, width, height = x.size()  # 8,28,8,8
        f = self.extra_projf(x).view(m_batchsize, -1, width * height)
        g = self.extra_projg(x).view(m_batchsize, -1, width * height)
        h = self.extra_projh(x).view(m_batchsize, -1, width * height)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = F.softmax(attention, dim=1)  # 8,1024,1024

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)  # B * C * W * H
        self_mask = self.extra_gate(self_attetion)  # [1, 1, 8, 8]
        self_mask = self.extra_gate_s(self_mask)
        out = self_mask * x
        return out

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