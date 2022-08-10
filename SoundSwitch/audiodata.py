import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageStat
import os
import h5py
from PIL import Image
import random
from torchvision import transforms

Audio_path = "F:\\train_audio3\\"
Forg_path = "G:\\AVE-ECCV18-master\\AVE_Dataset\\train_switch\\"  # audio  feature is true
def make_dataset(ori_path):
    path_listz = []
    path_listo = []
    count = 0
    ori_name = os.listdir(ori_path)
    for file in range(0, len(ori_name)):
        print(file)
        ficpath = os.path.join(ori_path, ori_name[file])
        ficname = os.listdir(ficpath)
        for fs in range(0, len(ficname)):
            picpath = os.path.join(ficpath, ficname[fs])
            picname = os.listdir(picpath)
            for picp in range(1, len(picname)-1):
                if os.path.exists(os.path.join(Forg_path, ori_name[file], ficname[fs], picname[picp][:-4]+'.jpg')):
                    onoroff = '1'
                    ps = os.path.join(picpath, picname[picp])
                    pa = os.path.join(Audio_path, ori_name[file], ficname[fs], picname[picp][:-4]+'_asp.h5')
                    path_listo.append(onoroff+'+'+pa+'+'+ps+'+'+str(file)+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
                else:
                    onoroff = '0'
                    ps = os.path.join(picpath, picname[picp])
                    pa = os.path.join(Audio_path, ori_name[file], ficname[fs], picname[picp][:-4]+'_asp.h5')
                    path_listz.append(onoroff+'+'+pa+'+'+ps+'+'+str(file)+'+'+ ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
    random.shuffle(path_listz)
    slice = random.sample(path_listz, len(path_listo))
    path_listo = path_listo+ slice
    return path_listo


class ImageFolder(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        pathimla = self.imgs[index]
        img_la = pathimla.split('+')
        onoroff = int(img_la[0])
        audio_path = img_la[1]
        video_path = img_la[2]

        with h5py.File(audio_path, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        audio_features_batch = torch.from_numpy(audio_features).float()
       
        video_features = Image.open(video_path).resize((356, 356), Image.ANTIALIAS).convert('RGB')  
        video_features_batch = self.img_transform(video_features)
        inda = int(img_la[-4])
        file = img_la[-3]
        subfile = img_la[-2]
        ssubfile = img_la[-1]

        return audio_features_batch, video_features_batch, onoroff, inda, file, subfile, ssubfile

    def __len__(self):
        return len(self.imgs)