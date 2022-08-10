import torch.nn as nn
import torch
import torch.nn.functional as F
affine_par = True
from ConvGRU import ConvGRUCell
import os
import torch.utils.model_zoo as model_zoo
from util import remove_layer

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
        self.features = features
        self.extra_convs = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True), 
                                         nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
                                         nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
                                         nn.Conv2d(512, 28, 1))

        self.extra_refineST = nn.Sequential(nn.Conv3d(28, 28, (3, 1, 1), padding=(0, 0, 0)),
                                            nn.Conv3d(28, 28, (1, 3, 3), padding=(0, 1, 1)),
                                            nn.Conv3d(28, 28, (1, 3, 3), padding=(0, 1, 1)))
        self.extra_s = nn.Sigmoid()

        self.extra_conv_stfusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias= True)

        self.extra_FRL = FRL(kwargs['refine_rate'], kwargs['refine_thresh'])

        self.channel = all_channel
        self.extra_projf = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projg = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel, out_channels=all_channel, kernel_size=1)
        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size = 1, bias = False)
        self.extra_gate_s = nn.Sigmoid()

        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.extra_conv_fusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias= True)
        self.extra_relu_fusion = nn.ReLU(inplace=True)

        self.propagate_layers = 3

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
    		
    def forward(self, idx, img_name0_0, img_name0_1, img_name0_2, img_name1_0, img_name1_1, img_name1_2, img_name2_0, img_name2_1, img_name2_2, 
            img0_0, img0_1, img0_2, img1_0, img1_1, img1_2, img2_0, img2_1, img2_2, epoch=1, label=None, index=None):

        batch_num  = img0_0.size()[0]

        x0_0 = self.features(img0_0)
        x0_0 = self.extra_convs(x0_0) # 1,28,32,32
        x0_0ss = F.avg_pool2d(x0_0,kernel_size=(x0_0.size(2),x0_0.size(3)),padding=0)
        x0_0ss = x0_0ss.view(-1, 28) # 2,28

        x0_1 = self.features(img0_1)
        x0_1 = self.extra_convs(x0_1) # 1,28,32,32
        self.map_0_1     = torch.zeros(batch_num, 28).cuda()
        self.map_0_1     = x0_1
        self.map_all_0_1 = torch.zeros(batch_num, 28, 32, 32).cuda()
        x0_1sss          = torch.zeros(batch_num, 28).cuda()
        x0_1ss = F.avg_pool2d(x0_1,kernel_size=(x0_1.size(2),x0_1.size(3)),padding=0)
        x0_1ss = x0_1ss.view(-1, 28) # 2,28

        x0_2 = self.features(img0_2)
        x0_2 = self.extra_convs(x0_2) # 1,28,32,32
        x0_2ss = F.avg_pool2d(x0_2,kernel_size=(x0_2.size(2),x0_2.size(3)),padding=0)
        x0_2ss = x0_2ss.view(-1, 28) # 2,28

        incat0 = torch.cat((x0_0.unsqueeze(1), x0_1.unsqueeze(1), x0_2.unsqueeze(1)), 1).view(batch_num,28,3,32,32)
        x0_1ST = self.extra_conv_stfusion(torch.cat((F.relu(x0_1+self.self_attention(x0_1)), F.relu(x0_1+x0_1*self.extra_s(self.extra_refineST(incat0).squeeze(2)))),1))
 
        x1_0 = self.features(img1_0) # 1,512,32,32
        x1_0 = self.extra_convs(x1_0) # 1,28,32,32
        x1_0ss = F.avg_pool2d(x1_0, kernel_size=(x1_0.size(2), x1_0.size(3)), padding=0) # 1,28,1,1
        x1_0ss = x1_0ss.view(-1, 28) # 1,28

        x1_1 = self.features(img1_1) # 1,512,32,32
        x1_1 = self.extra_convs(x1_1) # 1,28,32,32
        self.map_all_1_1 = torch.zeros(batch_num, 28, 32, 32).cuda()
        x1_1sss          = torch.zeros(batch_num, 28).cuda()
        x1_1ss = F.avg_pool2d(x1_1, kernel_size=(x1_1.size(2), x1_1.size(3)), padding=0) # 1,28,1,1
        x1_1ss = x1_1ss.view(-1, 28) # 1,28

        x1_2 = self.features(img1_2) # 1,512,32,32
        x1_2 = self.extra_convs(x1_2) # 1,28,32,32
        x1_2ss = F.avg_pool2d(x1_2, kernel_size=(x1_2.size(2), x1_2.size(3)), padding=0) # 1,28,1,1
        x1_2ss = x1_2ss.view(-1, 28) # 1,28

        incat1 = torch.cat((x1_0.unsqueeze(1), x1_1.unsqueeze(1), x1_2.unsqueeze(1)), 1).view(batch_num,28,3,32,32)
        x1_1ST = self.extra_conv_stfusion(torch.cat((F.relu(x1_1+self.self_attention(x1_1)), F.relu(x1_1+x1_1*self.extra_s(self.extra_refineST(incat1).squeeze(2)))),1))

        x2_0 = self.features(img2_0)
        x2_0 = self.extra_convs(x2_0) # 1,28,32,32
        x2_0ss = F.avg_pool2d(x2_0, kernel_size=(x2_0.size(2), x2_0.size(3)), padding=0)
        x2_0ss = x2_0ss.view(-1, 28)

        x2_1 = self.features(img2_1)
        x2_1 = self.extra_convs(x2_1) # 1,28,32,32
        self.map_all_2_1 = torch.zeros(batch_num, 28, 32, 32).cuda()
        x2_1sss          = torch.zeros(batch_num, 28).cuda()
        x2_1ss = F.avg_pool2d(x2_1, kernel_size=(x2_1.size(2), x2_1.size(3)), padding=0)
        x2_1ss = x2_1ss.view(-1, 28)

        x2_2 = self.features(img2_2)
        x2_2 = self.extra_convs(x2_2) # 1,28,32,32
        x2_2ss = F.avg_pool2d(x2_2, kernel_size=(x2_2.size(2), x2_2.size(3)), padding=0)
        x2_2ss = x2_2ss.view(-1, 28)
        incat2 = torch.cat((x2_0.unsqueeze(1), x2_1.unsqueeze(1), x2_2.unsqueeze(1)), 1).view(batch_num,28,3,32,32)
        x2_1ST = self.extra_conv_stfusion(torch.cat((F.relu(x2_1+self.self_attention(x2_1)), F.relu(x2_1+x2_1*self.extra_s(self.extra_refineST(incat2).squeeze(2)))),1))
        for ii in range(batch_num):
            exemplar = x0_1ST[ii, :, :, :][None].contiguous().clone()
            query    = x1_1ST[ii, :, :, :][None].contiguous().clone()
            query1   = x2_1ST[ii, :, :, :][None].contiguous().clone()
            for passing_round in range(self.propagate_layers):
                attention0 = self.extra_conv_fusion(torch.cat([self.generate_attention(exemplar, query),
                                         self.generate_attention(exemplar, query1)],1)) 
                attention1 = self.extra_conv_fusion(torch.cat([self.generate_attention(query, exemplar),
                                        self.generate_attention(query, query1)],1))
                attention2 = self.extra_conv_fusion(torch.cat([self.generate_attention(query1, exemplar),
                                        self.generate_attention(query1, query)],1))
                
                h_v0 = self.extra_FRL(attention0)
                h_v1 = self.extra_FRL(attention1)
                h_v2 = self.extra_FRL(attention2)
                
                exemplar = h_v0.clone()
                query = h_v1.clone()
                query1 = h_v2.clone()

                if passing_round == self.propagate_layers -1:
                    self.map_all_0_1[ii] = h_v0.clone()
                    x0s = F.avg_pool2d(h_v0, kernel_size=(h_v0.size(2), h_v0.size(3)), padding=0)
                    x0_1sss[ii] = x0s.view(-1, 28)  # 1,28,

                    self.map_all_1_1[ii] = h_v1.clone()
                    x1s = F.avg_pool2d(h_v1, kernel_size=(h_v1.size(2), h_v1.size(3)), padding=0)  # 1,28,1,1
                    x1_1sss[ii] = x1s.view(-1, 28) # 1,28

                    self.map_all_2_1[ii] = h_v2.clone()
                    x2s = F.avg_pool2d(h_v2, kernel_size=(h_v2.size(2), h_v2.size(3)), padding=0)
                    x2_1sss[ii] = x2s.view(-1, 28)  # 1,28,
       
        return x0_0ss,x0_1ss, x0_1sss,x0_2ss, x1_0ss,x1_1ss,x1_1sss,x1_2ss, x2_0ss,x2_1ss, x2_1sss,x2_2ss, self.map_0_1, self.map_all_0_1

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

cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'D2':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],}

def conservativeCGCN(pretrained=True, **kwargs):
    model = CGCN(make_layers(cfg['D1'], **kwargs), **kwargs)
    if pretrained:
        model = load_pretrained_model(model, path=None)
    return model
