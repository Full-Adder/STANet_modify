from __future__ import print_function
import os
import torch
import torch.nn as nn
import random
from models import att_Model
from logger import Logger
import time
import warnings
import argparse
from torch.utils.data import DataLoader
from audiodata import ImageFolder
warnings.filterwarnings("ignore")
random.seed(3344)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID
parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--model_name', type=str, default='AV_att',
                    help='model name')
parser.add_argument('--dir_order_train', type=str,
                    default='G:\\AVE-ECCV18-master\\AVE_Dataset\\train\\',
                    help='indices of training samples')
parser.add_argument('--nb_epoch', type=int, default=300,
                    help='number of epoch')
parser.add_argument('--batch_size', type=int, default=8, # 18
                    help='number of batch size')
parser.add_argument('--train', action='store_true', default=True,
                    help='train a new model')
args = parser.parse_args()

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

net_model = att_Model().cuda()

experiment_name = "debug1"
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
writer = Logger("output/logs/{}".format(experiment_name), 
                clear=True, port=8000, palette=palette)
# loss_function = nn.MultiLabelSoftMarginLoss()
loss_function = nn.CrossEntropyLoss().cuda()

def get_1x_lr_params(net_model):
    b = []
    b.append(net_model.layerA)
    b.append(net_model.layerV)
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(net_model):
    b = []
    b.append(net_model.layerA_d.parameters())
    b.append(net_model.layerV_d.parameters())
    b.append(net_model.extra_bilinear.parameters())
    b.append(net_model.fc.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

# optimizer = torch.optim.SGD(net_model.parameters(), lr=1e-2, momentum=0.9)
optimizer = torch.optim.SGD([{'params': get_1x_lr_params(net_model), 'lr': 1e-4},
                {'params': get_10x_lr_params(net_model), 'lr': 1e-4}], 
                lr=1e-3, momentum=0.9)

# optimizer = torch.optim.SGD(net_model.parameters(), lr=1e-2, momentum=0.9)


def main(args):
    full_dataset = ImageFolder(args.dir_order_train)
    train_size = int(0.99 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=0)
    epoch_l = []
    net_model.cuda().train()
    for epoch in range(args.nb_epoch):
        epoch_loss = 0
        n = 0
        for i, data in enumerate(train_loader):
            start = time.time()
            audio_inputs, video_inputs, onoroff, labela, file, subfile, ssubfile = data

            audio_inputs, video_inputs, labela, onoroff = audio_inputs.unsqueeze(1).cuda(), video_inputs.cuda(), labela.cuda(), onoroff.cuda()

            optimizer.zero_grad()
            scores = net_model(audio_inputs, video_inputs)
            loss2 = loss_function(scores, onoroff)
            loss = loss2
            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            optimizer.step()
            n = n + 1

            end = time.time()
            epoch_l.append(epoch_loss)
            if i % 1000 == 0:
                correct = 0
                test_total = 0
                with torch.no_grad():
                    for i, data_t in enumerate(test_loader):
                        net_model.eval()
                        audio_inputs, video_inputs, onoroff, labela, file, subfile, ssubfile = data_t
                        audio_inputs, video_inputs, labela, onoroff = audio_inputs.unsqueeze(1).cuda(), video_inputs.cuda(), labela.cuda(), onoroff.cuda()
                        scores1 = net_model(audio_inputs, video_inputs)
                        _, predicted = torch.max(scores1.data, 1)
                        correct += (predicted == onoroff.data).sum()
                        test_total += onoroff.size(0)
                print("=== Step {%s}   accuray: {%4f}" % (str(n), float(int(correct)/test_total)))
                torch.save(net_model, "./modelAV/"+ str(n) + ".pt")
                net_model.train()
            if i % 50 == 0:
                writer.add_scalar("loss", ((epoch_loss) / n).item(), i)
                writer.add_scalar("loss2", loss2.item(), i)

                print("=== Step {%s}   epoch_loss: {%.4f}  Loss: {%.4f}  Running time: {%4f}" % (str(n), (epoch_loss) / n, loss, end - start))

if __name__=="__main__":
    main(args)