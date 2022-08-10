import io
import os
import requests
import json
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.utils.data as data
import argparse
import cv2
import h5py
from model import AVENet
from torch.utils.data import DataLoader
import pdb
import csv
import string


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--summaries',
        default='vggsound_netvlad.pth.tar',
        # default='vggsound_avgpool.pth.tar',
        type=str,
        help='Directory path of pretrained model')
    parser.add_argument(
        '--pool',
        default="vlad",
        type=str,
        help='either vlad or avgpool')
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=309,
        type=int,
        help='Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    return parser.parse_args()


def make_dataset(root):
    path_list = []
    ori_name = os.listdir(root)
    for file in range(0, len(ori_name)):
        ficpath = os.path.join(root, ori_name[file])
        ficname = os.listdir(ficpath)
        for fs in range(0, len(ficname)):
            picpath = os.path.join(ficpath, ficname[fs])
            picname = os.listdir(picpath)
            for picp in range(0, len(picname)):
                pp = os.path.join(picpath, picname[picp])
                path_list.append(pp)
    return path_list


class ImageFolder(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        imgname = img_path.split('\\')
        with h5py.File('G:\\Data\\Audio3\\'+imgname[3]+'\\'+imgname[4]+'\\'+imgname[5][:-4]+'_asp.h5', 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])
        return torch.from_numpy(audio_features).float(), imgname[-3], imgname[-2], imgname[-1]

    def __len__(self):
        return len(self.imgs)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_arguments()
    model = AVENet(args).cuda().eval()
    # load pretrained models
    checkpoint = torch.load(args.summaries)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('load pretrained model.')
    model.eval()
    classes = []
    with open('data/stat.csv') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            classes.append(row[0])
    classes = sorted(classes)

    ori_path = 'G:\\Data\\Audio3\\'
    img2_path = 'G:\\Data\\val\\'
    positive_paht = 'G:\\Data\\val_switch\\'
    test_set = ImageFolder(img2_path)
    batch_size = 3
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             num_workers=0, shuffle=False)

    filename = 'fenlei.txt'
    t1 = 'Acoustic guitar'
    t2 = 'Baby cry, infant cry'
    t3 = 'Bark'
    t4 = 'Female speech, woman speaking'
    t5 = 'Fixed-wing aircraft, airplane'
    t6 = 'Frying (food)'
    t7 = 'Male speech, man speaking'
    t8 = 'Race car, auto racing'
    t9 = 'Rodents, rats, mice'
    t10 = 'Toilet flush'
    t11 = 'Train horn'
    t12 = 'Violin, fiddle'
    with torch.no_grad():
        with open(filename, 'w') as f:
            for num, [img_pil, ori_name, ficname, picname] in enumerate(test_loader):
                spec = Variable(img_pil.unsqueeze(1)).cuda()
                aud_o = model(spec.float())

                prediction = nn.Softmax(dim=1)(aud_o)
                probs, pred = torch.max(prediction.cpu().data, 1)
                for m in range(pred.size(0)):
                    print(classes[int(pred[m].numpy())])
                    t = ori_name[m]
                    s = classes[int(pred[m].numpy())]

                    if t == t1:
                        t = 'guitar'
                    if t == t2:
                        t = 'baby'
                    if t == t3:
                        t = 'dog'
                    if t == t4:
                        t = 'female'
                    if t == t5:
                        t = 'plane'
                    if t == t6:
                        t = 'food'
                    if t == t7:
                        t = 'male'
                    if t == t8:
                        t = 'car'
                    if t == t9:
                        t = 'mouse'
                    if t == t10:
                        t = 'toilet'
                    if t == t11:
                        t = 'train'
                    if t == t12:
                        t = 'Violin'
                    t = t.lower()
                    result = str.find(s, t) != -1
                    if result:
                        positive_savepaht = positive_paht+'\\' + \
                            ori_name[m]+'\\'+ficname[m]+'\\'
                        if not os.path.exists(positive_savepaht):
                            os.makedirs(positive_savepaht)
                        xmm = cv2.resize(cv2.imread(
                            img2_path+'\\'+ori_name[m]+'\\'+ficname[m]+'\\'+picname[m]),(1,1))
                        cv2.imwrite(positive_savepaht +
                                    picname[m], xmm)
                    '''else: 
                        negative_savepath = negative_path+'\\' + \
                            ori_name[m]+'\\'+ficname[m]+'\\'
                        if not os.path.exists(negative_savepath):
                            os.makedirs(negative_savepath)
                        xmm = cv2.imread(
                            img2_path+'\\'+ori_name[m]+'\\'+ficname[m]+'\\'+picname[m][:-7]+'.jpg')
                        cv2.imwrite(negative_savepath+picname[m][:-7]+'.jpg', xmm)'''

                

if __name__ == "__main__":
    main()
