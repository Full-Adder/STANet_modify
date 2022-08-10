from Soundmodel import SoundNet
from ConvGRU import ConvGRUCell
import torch.nn as nn
import torch
import torch.nn.functional as F
affine_par = True

class STANet(nn.Module):
    def __init__(self, all_channel=28, att_dir='./runs/', training_epoch=10, **kwargs):
        super(STANet, self).__init__()
        self.extra_audio_d = nn.Linear(8192, 2048)
        self.extra_bilinear = nn.Bilinear(144, 1, 144)
        self.Aup = nn.Sequential(nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=0),
                                nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=0),
                                nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=0))

        self.extra_video_d0 = nn.Conv2d(512, 28, 1)
        self.extra_video_d1 = nn.Conv2d(1024, 28, 1)
        self.extra_video_d2 = nn.Conv2d(2048, 28, 1)
        self.extra_convs = nn.Sequential(nn.Conv2d(2048, 28, 1), nn.Conv2d(28, 1, 1),nn.Sigmoid()) 
        self.s = nn.Sigmoid()
        
        all_channel = 28
        self.extra_conv_fusion = nn.Conv2d(all_channel*3, all_channel, kernel_size=1, bias=True)
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.extra_gate_s = nn.Sigmoid()
        self.extra_projf = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projg = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel, out_channels=all_channel, kernel_size=1)
        self.extra_refineST = nn.Conv3d(28, 1, (3, 1, 1), padding=(0, 0, 0))

        self.extra_refine4 = nn.Conv2d(56, 28, kernel_size=3, padding=1)
        self.extra_refine3 = nn.Conv2d(56, 28, kernel_size=3, padding=1)
        self.extra_refine2 = nn.Conv2d(56, 28, kernel_size=3, padding=1)
        self.extra_predict4 = nn.Conv2d(28, 1, 1)
        self.extra_predict3 = nn.Conv2d(28, 1, 1)
        self.extra_predict2 = nn.Conv2d(28, 1, 1)

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

        net = torch.hub.load('facebookresearch/WSL-Images',
                             'resnext101_32x8d_wsl')
        net = list(net.children())
        self.features0 = nn.Sequential(*net[:6])
        self.features1 = nn.Sequential(*net[6])
        self.features2 = nn.Sequential(*net[7])
        

    def forward(self, idx,  img_name1, img_name2, img_name3, input1, input2, input3, audio1, audio2, audio3, switch1, switch2, switch3, epoch=1, label=None, index=None):

        batch_num = input1.size()[0]
        input_size = input1.size()[2:]

        a1  = self.audio_model(audio1.unsqueeze(1))  # [2, 8192, 1]
        a1  = self.extra_audio_d(a1).unsqueeze(2)  # [2, 2048, 1]
        a2  = self.audio_model(audio2.unsqueeze(1))
        a2  = self.extra_audio_d(a2).unsqueeze(2)
        a3  = self.audio_model(audio3.unsqueeze(1))
        a3  = self.extra_audio_d(a3).unsqueeze(2)

        x10 = self.features0(input1)    # 2,512,45,45
        x11 = self.features1(x10)       # 2,1024,23,23
        x1  = self.features2(x11)        # 2,2048,12,12
        x20 = self.features0(input2)
        x21 = self.features1(x20)
        x2  = self.features2(x21)
        x30 = self.features0(input3)
        x31 = self.features1(x30)
        x3  = self.features2(x31)
        
        Aup1 = self.Aup(a1.unsqueeze(2))# 1,2048,8,8
        Aup2 = self.Aup(a2.unsqueeze(2))# 1,2048,8,8
        Aup3 = self.Aup(a3.unsqueeze(2))# 1,2048,8,8

        x10 = self.extra_video_d0(x10)  # 1,28,28,28
        x11 = self.extra_video_d1(x11)  # 1,8,28,28
        x1  = self.extra_video_d2(x1)    # 1,28,8,8
        x20 = self.extra_video_d0(x20)
        x21 = self.extra_video_d1(x21)
        x2  = self.extra_video_d2(x2)       
        x30 = self.extra_video_d0(x30)
        x31 = self.extra_video_d1(x31)
        x3  = self.extra_video_d2(x3)   
        
        incat1 = torch.cat((x2.unsqueeze(1), x1.unsqueeze(1), x3.unsqueeze(1)), 1).view(batch_num,28,3,12,12)
        incat2 = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)), 1).view(batch_num,28,3,12,12)
        incat3 = torch.cat((x1.unsqueeze(1), x3.unsqueeze(1), x2.unsqueeze(1)), 1).view(batch_num,28,3,12,12)

        x12       = self.extra_conv_fusion(torch.cat((F.relu(x1+self.self_attention(x1)), F.relu(x1+self.s(self.extra_refineST(incat1).squeeze(2))), F.relu(x1+x1*self.extra_convs(switch1*Aup1))), 1))
        refine04  = self.extra_refine4(torch.cat((x12, x1), 1))
        refine04  = F.upsample(refine04, size=x11.size()[2:], mode='bilinear')
        refine03  = self.extra_refine3(torch.cat((refine04, x11), 1))
        refine03  = F.upsample(refine03, size=x10.size()[2:], mode='bilinear')
        refine02  = self.extra_refine2(torch.cat((refine03, x10), 1))
        predict04 = self.extra_predict4(refine04)
        predict03 = self.extra_predict3(refine03)
        predict02 = self.extra_predict2(refine02)
        predict04 = F.upsample(predict04, size=input_size, mode='bilinear')
        predict03 = F.upsample(predict03, size=input_size, mode='bilinear')
        predict02 = F.upsample(predict02, size=input_size, mode='bilinear')

        x22       = self.extra_conv_fusion(torch.cat((F.relu(x2+self.self_attention(x2)), F.relu(x2+self.s(self.extra_refineST(incat2).squeeze(2))), F.relu(x2+self.extra_convs(switch2*Aup2))), 1))
        refine14  = self.extra_refine4(torch.cat((x22, x2), 1))
        refine14  = F.upsample(refine14, size=x21.size()[2:], mode='bilinear')
        refine13  = self.extra_refine3(torch.cat((refine14, x21), 1))
        refine13  = F.upsample(refine13, size=x20.size()[2:], mode='bilinear')
        refine12  = self.extra_refine2(torch.cat((refine13, x20), 1))
        predict14 = self.extra_predict4(refine14)
        predict13 = self.extra_predict3(refine13)
        predict12 = self.extra_predict2(refine12)
        predict14 = F.upsample(predict14, size=input_size, mode='bilinear')
        predict13 = F.upsample(predict13, size=input_size, mode='bilinear')
        predict12 = F.upsample(predict12, size=input_size, mode='bilinear')

        x33       = self.extra_conv_fusion(torch.cat((F.relu(x3+self.self_attention(x3)), F.relu(x3+self.s(self.extra_refineST(incat3).squeeze(2))), F.relu(x3+self.extra_convs(switch3*Aup3))), 1))
        refine24  = self.extra_refine4(torch.cat((x33, x3), 1))
        refine24  = F.upsample(refine24, size=x31.size()[2:], mode='bilinear')
        refine23  = self.extra_refine3(torch.cat((refine24, x31), 1))
        refine23  = F.upsample(refine23, size=x30.size()[2:], mode='bilinear')
        refine22  = self.extra_refine2(torch.cat((refine23, x30), 1))

        predict24 = self.extra_predict4(refine24)
        predict23 = self.extra_predict3(refine23)
        predict22 = self.extra_predict2(refine22)
        predict24 = F.upsample(predict24, size=input_size, mode='bilinear')
        predict23 = F.upsample(predict23, size=input_size, mode='bilinear')
        predict22 = F.upsample(predict22, size=input_size, mode='bilinear')

        return predict04,predict03,predict02, predict14,predict13,predict12, predict24,predict23,predict22

    def self_attention(self, x):
        m_batchsize, C, width, height = x.size()
        f = self.extra_projf(x).view(m_batchsize, -1, width * height)
        g = self.extra_projg(x).view(m_batchsize, -1, width * height)
        h = self.extra_projh(x).view(m_batchsize, -1, width * height)

        attention     = torch.bmm(f.permute(0, 2, 1), g)
        attention     = F.softmax(attention, dim=1)

        self_attetion = torch.bmm(h, attention)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)
        self_mask     = self.extra_gate(self_attetion)
        self_mask     = self.extra_gate_s(self_mask)
        out           = self_mask * x
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
