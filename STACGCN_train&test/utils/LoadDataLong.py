from .transforms import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import h5py

def train_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225] 
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])
    tsfm_grount = transforms.Compose([transforms.Resize(crop_size), transforms.ToTensor()])
    tsfm_test = transforms.Compose([ transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDatasetT(audioroot_dir=args.audio_dir1, swtichroot_dir=args.swtich_dir1, imgroot_dir=args.img_dir1, groundt_dir=args.groundt_dir1, num_classes=args.num_classes, transform_im=tsfm_train, transform_gt=tsfm_grount, test=False)
    img_test  = VOCDatasetE(audioroot_dir=args.audio_dir2, swtichroot_dir=args.swtich_dir2, imgroot_dir=args.img_dir2, groundt_dir=args.groundt_dir2, num_classes=args.num_classes, transform_im=tsfm_test, transform_gt=tsfm_grount, test=True)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(img_test,  batch_size=3, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def test_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])
    tsfm_grount = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])

    img_test  = VOCDatasetE(audioroot_dir=args.audio_dir2, swtichroot_dir=args.swtich_dir2, imgroot_dir=args.img_dir2, groundt_dir=args.groundt_dir2, num_classes=args.num_classes, transform_im=tsfm_test, transform_gt=tsfm_grount, test=True)
    val_loader   = DataLoader(img_test,  batch_size=64, shuffle=False, num_workers=args.num_workers)

    return val_loader


class VOCDatasetT(Dataset):
    def __init__(self, audioroot_dir, swtichroot_dir, imgroot_dir, groundt_dir, num_classes=28, transform_im=None, transform_gt=None, test=False):
        self.audioroot_dir = audioroot_dir
        self.groundt_dir = groundt_dir
        self.swtichroot_dir = swtichroot_dir
        self.imgroot_dir = imgroot_dir
        self.testing = test
        self.transform_im = transform_im
        self.transform_gt = transform_gt
        self.num_classes = num_classes
        self.image_list, self.name_list, self.figs_list, self.seqs_list = self.read_labeled_image_list_train(self.audioroot_dir,self.imgroot_dir,self.swtichroot_dir, self.groundt_dir)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):    
        pathimla = self.image_list[idx]
        img_la = pathimla.split('+')

        inda = int(img_la[3])
        pathname = self.name_list[inda]
        images = []
        images2= []

        audios = []
        image_names = []
        groundts = []

        for hh in range(0, 3):
            aa = self.seqs_list[inda].index(img_la[8])  # 找sequence
            bb = random.randrange(1, pathname[aa][1]) # 找frame
           
            # 相邻3帧
            for jj in range(0, 3):
                pick_path = os.path.join(img_la[5], img_la[7], pathname[aa][0], '%04d'%(int(self.figs_list[inda][aa][bb][:-4])+jj)+'.jpg')
                image_pick = Image.open(pick_path).convert('RGB')
                image_pick = self.transform_im(image_pick.resize((256, 256), Image.ANTIALIAS))
                image_names.append(pick_path)
                images.append(image_pick)
                if jj == 1:
                    Apick_path = os.path.join(img_la[4], img_la[7], pathname[aa][0], '%04d'%(int(self.figs_list[inda][aa][bb][:-4])+jj)+'_asp.h5')
                    with h5py.File(Apick_path, 'r') as hf:
                        audio_features = np.float32(hf['dataset'][:])  # 5,128
                    audio_pick = torch.from_numpy(audio_features).float()
                    audios.append(audio_pick)

                    Gpick_path = os.path.join(img_la[6], img_la[7], pathname[aa][0], '%04d'%(int(self.figs_list[inda][aa][bb][:-4])+jj)+'_c.jpg')
                    ground_pick = Image.open(Gpick_path).convert('L')
                    ground_pick = self.transform_gt(ground_pick)
                    groundts.append(ground_pick)

                    image_pick2 = Image.open(pick_path).convert('RGB')
                    image_pick2 = self.transform_im(image_pick2.resize((356, 356), Image.ANTIALIAS))
                    images2.append(image_pick2)
            
        label = np.zeros(28, dtype=np.float32)
        label[inda] = 1
        return  image_names[0], images[0], image_names[1], images[1], image_names[2], images[2], \
                image_names[4], images[4], image_names[3], images[3], image_names[5], images[5], \
                image_names[7], images[7], image_names[6], images[6], image_names[8], images[8], \
                audios[0], audios[1], audios[2], \
                groundts[0], groundts[1], groundts[2], \
                images2[0], images2[1], images2[2], \
                inda, label


    def read_labeled_image_list_train(self, audio_dir, data_dir, swtich_dir, groundt_dir):
        path_list = []
        names_list = []
        figs_list = []
        seqs_list = []
        ori_name = os.listdir(groundt_dir)
        for file in range(0, len(ori_name)):
            print(file)
            ficpath = os.path.join(groundt_dir, ori_name[file])
            ficname = os.listdir(ficpath)
            name_list = []
            fig_list = []
            seq_list = []
            ficnames = []
            for fs in range(0, len(ficname)):
                picpath = os.path.join(ficpath, ficname[fs])
                picname = os.listdir(picpath)
                if len(picname) < 9:
                    continue
                picnames = []
                if os.path.exists(os.path.join(groundt_dir, ori_name[file], ficname[fs])):
                    for picp in range(3, len(picname)-3):
                        if picname[picp].endswith('_c.jpg'):
                            if os.path.exists(os.path.join(groundt_dir, ori_name[file], ficname[fs], '%04d' % (int(picname[picp][:-6]))+'_c.jpg')) and os.path.exists(os.path.join(groundt_dir, ori_name[file], ficname[fs], '%04d' % (int(picname[picp][:-6])+1)+'_c.jpg')):
                                pv = os.path.join(data_dir, ori_name[file], ficname[fs], picname[picp][:-6]+'.jpg')
                                pa = os.path.join(audio_dir, ori_name[file], ficname[fs], picname[picp][:-6]+'_asp.h5')
                                ps = os.path.join(groundt_dir, ori_name[file], ficname[fs], picname[picp][:-6]+'_c.jpg')
                                picnames.append(picname[picp][:-6]+'.jpg')
                                ficnames = ficname[fs]
                                path_list.append(pv+'+'+pa+'+'+ps+'+'+str(file)+'+'+audio_dir+'+'+data_dir+'+'+groundt_dir+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-6]+'.jpg')
                    name_list.append([ficnames, len(picnames)])
                    fig_list.append(picnames)
                    seq_list.append(ficnames)
            names_list.append(name_list)
            figs_list.append(fig_list)
            seqs_list.append(seq_list)
        return path_list, names_list, figs_list, seqs_list

class VOCDatasetE(Dataset):
    def __init__(self, audioroot_dir, swtichroot_dir, imgroot_dir, groundt_dir, num_classes=28, transform_im=None, transform_gt=None, test=False):
        self.audioroot_dir = audioroot_dir
        self.groundt_dir = groundt_dir
        self.swtichroot_dir = swtichroot_dir
        self.imgroot_dir = imgroot_dir
        self.testing = test
        self.transform_im = transform_im
        self.transform_gt = transform_gt
        self.num_classes = num_classes
        self.image_list, self.name_list, self.figs_list, self.seqs_list = self.read_labeled_image_list_test(self.audioroot_dir,self.imgroot_dir,self.swtichroot_dir, self.groundt_dir)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):    
        pathimla = self.image_list[idx]
        img_la = pathimla.split('+')

        inda = int(img_la[2])
        pathname = self.name_list[inda]
        images = []
        audios = []
        image_names = []
        images2 = []
        for hh in range(0, 3):
            aa = self.seqs_list[inda].index(img_la[6])  # 找sequence
            bb = random.randrange(1, pathname[aa][1]) # 找frame

            # 相邻3帧
            for jj in range(0, 3):
                pick_path = os.path.join(img_la[4], img_la[5], pathname[aa][0], 'img_%05d'%(int(self.figs_list[inda][aa][bb][4:-4])+jj)+'.jpg')
                image_pick = Image.open(pick_path).convert('RGB')
                image_pick = self.transform_im((image_pick.resize((256, 256), Image.ANTIALIAS)))
                image_names.append(pick_path)
                images.append(image_pick)
                if jj == 1:
                    Apick_path = os.path.join(img_la[3], img_la[5], pathname[aa][0], self.figs_list[inda][aa][bb][:-4]+'_asp.h5')
                    with h5py.File(Apick_path, 'r') as hf:
                        audio_features = np.float32(hf['dataset'][:])  # 5,128
                    audio_pick = torch.from_numpy(audio_features).float()
                    audios.append(audio_pick)

                    image_pick2 = Image.open(pick_path).convert('RGB')
                    image_pick2 = self.transform_im(image_pick2.resize((356, 356), Image.ANTIALIAS))
                    images2.append(image_pick2)

        label = np.zeros(28, dtype=np.float32)
        label[inda] = 1
        return  image_names[0], images[0], image_names[1], images[1], image_names[2], images[2], \
                image_names[3], images[3], image_names[4], images[4], image_names[5], images[5], \
                image_names[6], images[6], image_names[7], images[7], image_names[8], images[8], \
                audios[0], audios[1], audios[2], \
                images2[0], images2[1], images2[2], \
                inda, label

    def read_labeled_image_list_test(self, audio_dir, data_dir, swtich_dir, groundt_dir):
        path_list = []
        names_list = []
        figs_list = []
        seqs_list = []
        ficnames = []
        ori_name = os.listdir(data_dir)
        ori_name.sort()
        for file in range(0, 1):#len(ori_name)):
            print(file)
            ficpath = os.path.join(data_dir, ori_name[file])
            ficname = os.listdir(ficpath)
            ficname.sort()
            name_list = []
            fig_list = []
            seq_list = []
            for fs in range(0, len(ficname)):
                picpath = os.path.join(ficpath, ficname[fs])
                picname = os.listdir(picpath)
                picname.sort()
                if len(picname) < 3:
                    continue
                picnames = []
                xmmhh = []
                for xmm in range(0, len(picname)):
                    if picname[xmm].endswith('.jpg'):
                        xmmhh.append(picname[xmm])
                picname = xmmhh
                for picp in range(6, len(picname)-6):
                    if picname[picp].endswith('.jpg'):
                        # switchpa = os.path.join(swtich_dir, ori_name[file], ficname[fs], picname[picp])
                        # if os.path.exists(switchpa):
                        pv = os.path.join(data_dir, ori_name[file], ficname[fs], picname[picp])
                        pa = os.path.join(audio_dir, ori_name[file], ficname[fs], picname[picp][:-4]+'_asp.h5')
                        picnames.append(picname[picp])
                        ficnames = ficname[fs]
                        path_list.append(pv+'+'+pa+'+'+str(file)+'+'+audio_dir+'+'+data_dir+'+'+ori_name[file]+'+'+ficname[fs]+'+'+picname[picp][:-4]+'.jpg')
                name_list.append([ficnames, len(picnames)])
                fig_list.append(picnames)
                seq_list.append(ficnames)
            names_list.append(name_list)
            figs_list.append(fig_list)
            seqs_list.append(seq_list)
        return path_list, names_list, figs_list, seqs_list

    
