import torch.nn as nn
import torch
import torch.nn.functional as F
affine_par = True
from ConvGRU import ConvGRUCell
import os
import torch.utils.model_zoo as model_zoo
from util import remove_layer
from Soundmodel import SoundNet

model_urls = {'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class FRL(nn.Module):
    def __init__(self, refine_rate, refine_thresh):
        super(FRL, self).__init__()
        self.refine_rate = refine_rate
        self.refine_thresh = refine_thresh

    def forward(self, input_):
        cMean = torch.mean(input_, dim=1, keepdim=True) # 1,1,32,32
        
        batch_size = cMean.size(0)
        maxval, _ = torch.max(cMean.view(batch_size, -1), dim=1, keepdim=True)
        throld = maxval * self.refine_thresh
        throld = throld.view(batch_size, 1, 1, 1)

        rMask = (cMean < throld).float()
        cMean = torch.sigmoid(cMean)
        attention = self.refine_rate * cMean * rMask + cMean * (1-rMask)
        return (input_.mul(attention) + input_) / 2


class CGCN(nn.Module):
    def  __init__(self, features, all_channel=28, att_dir='./runs/', training_epoch=10,**kwargs):
        super(CGCN, self).__init__()
        
        self.extra_audio_d = nn.Linear(8192, 512)
        self.extra_bilinear = nn.Bilinear(1024, 1, 1024)

        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 28, 1)
        )
        self.channel = all_channel
        self.extra_conv_safusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias= True)
        self.extra_projf = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projg = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel, out_channels=all_channel, kernel_size=1)
        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size = 1, bias = False)
        self.extra_gate_s = nn.Sigmoid()
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.extra_refineSA = nn.Sequential(nn.Conv2d(512, all_channel, kernel_size=1), nn.Conv2d(all_channel, 1, 1), nn.Sigmoid())
        self.extra_conv_fusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias= True)
        self.extra_relu_fusion = nn.ReLU(inplace=True)
        self.softmax = nn.Sigmoid()
        self.propagate_layers = 3

        self.extra_FRL = FRL(kwargs['refine_rate'], kwargs['refine_thresh'])

        d = self.channel // 2
        self.extra_proja = nn.Conv2d(self.channel, d, kernel_size=1)
        self.extra_projb = nn.Conv2d(self.channel, d, kernel_size=1)

        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir) 

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
    		
    def forward(self, idx,  img_name1, img_name2, img_name3, input1, input2, input3, audio1, audio2, audio3, epoch=1, label=None, index=None):

        batch_num  = input1.size()[0]

        a1 = self.audio_model(audio1.unsqueeze(1)) # [13, 8192]
        a1 = self.extra_audio_d(a1).unsqueeze(2) # [13, 512]     
        x1 = self.features(input1) # 1,512,32,32
        av1 = self.extra_bilinear(x1.contiguous().flatten(2), a1).view(x1.size(0), x1.size(1), x1.size(2), x1.size(3))
        x1 = self.extra_convs(x1) # 1,28,32,32
        x1 = self.extra_conv_safusion(torch.cat((F.relu(x1+self.self_attention(x1)), F.relu(x1+self.extra_refineSA(av1))), 1)) # 1,28,32,32
        self.map_1 = torch.zeros(batch_num,28,32,32).cuda()
        self.map_all_1 = torch.zeros(batch_num,28,32,32).cuda()
        self.map_1 = x1
        x1sss = torch.zeros(batch_num,28).cuda()
        x1ss = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0) # 1,28,1,1
        x1ss = x1ss.view(-1, 28) # 1,28
        
        a2 = self.audio_model(audio2.unsqueeze(1))
        a2 = self.extra_audio_d(a2).unsqueeze(2)
        x2 = self.features(input2)
        av2 = self.extra_bilinear(x2.contiguous().flatten(2), a2).view(x2.size(0), x2.size(1), x2.size(2), x2.size(3))
        x2 = self.extra_convs(x2) # 1,28,32,32
        x2 = self.extra_conv_safusion(torch.cat((F.relu(x2+self.self_attention(x2)), F.relu(x2+self.extra_refineSA(av2))), 1)) # 1,28,32,32
        self.map_2 = torch.zeros(batch_num,28,32,32).cuda()
        self.map_all_2 = torch.zeros(batch_num,28,32,32).cuda()
        self.map_2 = x2
        x2sss = torch.zeros(batch_num,28).cuda()
        x2ss = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        x2ss = x2ss.view(-1, 28)

        a3 = self.audio_model(audio3.unsqueeze(1))
        a3 = self.extra_audio_d(a3).unsqueeze(2)
        x3 = self.features(input3)
        av3 = self.extra_bilinear(x3.contiguous().flatten(2), a3).view(x3.size(0), x3.size(1), x3.size(2), x3.size(3))
        x3 = self.extra_convs(x3) # 1,28,32,32
        x3 = self.extra_conv_safusion(torch.cat((F.relu(x3+self.self_attention(x3)), F.relu(x3+self.extra_refineSA(av3))), 1)) # 1,28,32,32
        self.map_3 = torch.zeros(batch_num,28,32,32).cuda()
        self.map_all_3 = torch.zeros(batch_num,28,32,32).cuda()
        self.map_3 = x3
        x3sss = torch.zeros(batch_num,28).cuda()
        x3ss = F.avg_pool2d(x3,kernel_size=(x3.size(2),x3.size(3)),padding=0)
        x3ss = x3ss.view(-1, 28) # 2,28

        for ii in range(batch_num):
            exemplar = x1[ii,:,:,:][None].contiguous().clone()
            query    = x2[ii,:,:,:][None].contiguous().clone()
            query1   = x3[ii,:,:,:][None].contiguous().clone()
            for passing_round in range(self.propagate_layers):
                attention1 = self.extra_conv_fusion(torch.cat([self.generate_attention(exemplar, query),
                                         self.generate_attention(exemplar, query1)],1)) 
                attention2 = self.extra_conv_fusion(torch.cat([self.generate_attention(query, exemplar),
                                        self.generate_attention(query, query1)],1))
                attention3 = self.extra_conv_fusion(torch.cat([self.generate_attention(query1, exemplar),
                                        self.generate_attention(query1, query)],1))
                
                h_v1 = self.extra_ConvGRU(attention1, exemplar)
                h_v2 = self.extra_ConvGRU(attention2, query)
                h_v3 = self.extra_ConvGRU(attention3, query1)

                h_v1 = self.extra_FRL(h_v1)
                h_v2 = self.extra_FRL(h_v2)
                h_v3 = self.extra_FRL(h_v3)
                
                exemplar = h_v1.clone()
                query    = h_v2.clone()
                query1   = h_v3.clone()

                if passing_round == self.propagate_layers -1:
                    
                    self.map_all_1[ii] = h_v1.clone()
                    x1s = F.avg_pool2d(h_v1, kernel_size=(h_v1.size(2), h_v1.size(3)), padding=0)  # 1,28,1,1
                    x1sss[ii] = x1s.view(-1, 28) # 1,28

                    self.map_all_2[ii] = h_v2.clone()
                    x2s = F.avg_pool2d(h_v2, kernel_size=(h_v2.size(2), h_v2.size(3)), padding=0)
                    x2sss[ii] = x2s.view(-1, 28)  # 1,28,

                    self.map_all_3[ii] = h_v3.clone()
                    x3s = F.avg_pool2d(h_v3, kernel_size=(h_v3.size(2), h_v3.size(3)), padding=0)
                    x3sss[ii] = x3s.view(-1, 28)  # 1,28,
       
        return x1ss,x1sss, x2ss,x2sss, x3ss,x3sss, self.map_1, self.map_all_1
    
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

    def message_fun(self,input):
        input1 = self.extra_conv_fusion(input)
        input1 = self.extra_relu_fusion(input1)
        return input1

    def generate_attention(self, exemplar, query):
        fea_size = query.size()[2:]                                                     # 32,32
        N1, C1, H1, W1 = exemplar.shape                                                 # 1,28,32,32
        exemplar_low = self.extra_proja(exemplar)                                       # 1,14,32,32
        query_low = self.extra_projb(query)                                             # 1,14,32,32
        N,C,H,W = exemplar_low.shape                                                    # 1,14,32,32

        exemplar_flat = exemplar_low.view(N, C, H*W)                                    #[1, 14, 1024]
        query_flat = query_low.view(N, C, H*W)                                          #[1, 14, 1024]
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()                    #[1, 1024, 14]

        A = torch.bmm(exemplar_t, query_flat)                                           #[1, 1024, 1024]
        B = F.softmax(torch.transpose(A,1,2),dim=1)                                     #[1, 1024, 1024]
       
        exemplar_ = exemplar.view(N1, C1, H1 * W1)                                      #[1, 28, 1024]
        query_ = query.view(N1, C1, H1 * W1)                                            #[1, 28, 1024]

        exemplar_att = torch.bmm(query_, B).contiguous()                                #[1, 28, 1024]
        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])      #[1, 28, 32, 32]
        input1_mask = self.extra_gate(input1_att)                                       #[1, 1, 32, 32]
        input1_mask = self.extra_gate_s(input1_mask)                                    #[1, 1, 32, 32]
        input1_att = input1_att * input1_mask
        return input1_att                                                               #[1, 28, 32, 32]
 
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

def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model
   
def load_pretrained_model(model, path=None):
    state_dict = model_zoo.load_url(model_urls['vgg16'], progress=True)
    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    return model

def make_layers(cfg, batch_norm=False,**kwargs):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)            
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'D2':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def conservativeCGCN(pretrained=True, **kwargs):
    model = CGCN(make_layers(cfg['D1'], **kwargs), **kwargs)
    if pretrained:
        model = load_pretrained_model(model, path=None)
    return model
