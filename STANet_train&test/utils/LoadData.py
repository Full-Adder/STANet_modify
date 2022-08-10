from .transforms import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import h5py


def train_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225] 
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([transforms.Resize(crop_size),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])
    tsfm_grount = transforms.Compose([transforms.Resize(crop_size), transforms.ToTensor()])

    tsfm_test = transforms.Compose([transforms.Resize(crop_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDatasetT(audioroot_dir=args.audio_dir1, swtichroot_dir=args.swtich_dir1, imgroot_dir=args.img_dir1, groundt_dir=args.groundt_dir1, num_classes=args.num_classes, transform_im=tsfm_train, transform_gt=tsfm_grount, test=False)
    img_test  = VOCDatasetE(audioroot_dir=args.audio_dir2, swtichroot_dir=args.swtich_dir2, imgroot_dir=args.img_dir2, groundt_dir=args.groundt_dir2, num_classes=args.num_classes, transform_im=tsfm_test, transform_gt=tsfm_grount, test=True)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(img_test,  batch_size=2, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def test_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])
    tsfm_grount = transforms.ToTensor()
    img_test = VOCDatasetE(audioroot_dir=args.audio_dir2, swtichroot_dir=args.swtich_dir2, imgroot_dir=args.img_dir2, groundt_dir=args.groundt_dir2, num_classes=args.num_classes, transform_im=tsfm_test, transform_gt=tsfm_grount, test=True)
    val_loader = DataLoader(img_test, batch_size=2, shuffle=False, num_workers=args.num_workers)

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

        img_name = img_la[0]
        image = Image.open(img_name).convert('RGB')
        image = self.transform_im(image)

        audio_name = img_la[1]
        with h5py.File(audio_name, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        audio = torch.from_numpy(audio_features).float()

        gt_name = img_la[2]
        groundt = Image.open(gt_name).convert('L')
        groundt = self.transform_gt(groundt)

        inda = int(img_la[3])
        pathname = self.name_list[inda]
        images = []
        audios = []
        groundts = []
        image_names = []
        for hh in range(-1, 2):
            Vpick_path = os.path.join(img_la[5], img_la[7], img_la[8], '%04d'%(int(img_la[9][:-4])+hh)+'.jpg')
            Apick_path = os.path.join(img_la[4], img_la[7], img_la[8], '%04d'%(int(img_la[9][:-4])+hh)+'_asp.h5')  
            Gpick_path = os.path.join(img_la[6], img_la[7], img_la[8], '%04d'%(int(img_la[9][:-4])+hh)+'_c.jpg')    

            image_pick = Image.open(Vpick_path).convert('RGB')
            image_pick = self.transform_im(image_pick)
            image_names.append(Vpick_path)
            images.append(image_pick)

            ground_pick = Image.open(Gpick_path).convert('L')
            ground_pick = self.transform_gt(ground_pick)
            groundts.append(ground_pick)

            with h5py.File(Apick_path, 'r') as hf:
                audio_features = np.float32(hf['dataset'][:])  # 5,128
            audio_pick = torch.from_numpy(audio_features).float()
            audios.append(audio_pick)
            
        label = np.zeros(28, dtype=np.float32)
        label[inda] = 1
        return img_name, image, audio, inda, label, groundt, image_names[0], images[0], audios[0], inda, label, groundts[0], image_names[2], images[2], audios[2], inda, label, groundts[2]
    
    def read_labeled_image_list_train(self, audio_dir, data_dir, swtich_dir, groundt_dir):
        path_list = []
        names_list = []
        figs_list = []
        seqs_list = []
        ori_name = os.listdir(groundt_dir)
        for file in range(0, 1):#len(ori_name)):
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
                if len(picname) < 3:
                    continue
                picnames = []
                if os.path.exists(os.path.join(groundt_dir, ori_name[file], ficname[fs])):
                    for picp in range(2, len(picname)-2):
                        if picname[picp].endswith('_c.jpg'):
                            if (os.path.exists(os.path.join(picpath, str(int(picname[picp][:-6])-1).zfill(4)+'_c.jpg')) and os.path.exists(os.path.join(picpath, str(int(picname[picp][:-6])).zfill(4)+'_c.jpg')) and os.path.exists(os.path.join(picpath, str(int(picname[picp][:-6])+1).zfill(4)+'_c.jpg'))):
                                pv = os.path.join(data_dir, ori_name[file], ficname[fs], picname[picp][:-6]+'.jpg')
                                pa = os.path.join(audio_dir, ori_name[file], ficname[fs], picname[picp][:-6]+'_asp.h5')
                                ps = os.path.join(groundt_dir, ori_name[file], ficname[fs], picname[picp][:-6]+'_c.jpg')
                                picnames.append(picname[picp])
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

        img_name = img_la[0]
        image = Image.open(img_name).convert('RGB')
        image = self.transform_im(image)

        audio_name = img_la[1]
        with h5py.File(audio_name, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        audio = torch.from_numpy(audio_features).float()

        inda = int(img_la[2])
        images = []
        audios = []
        image_names = []
        for hh in range(-1,2):
            Vpick_path = os.path.join(img_la[4], img_la[5], img_la[6], 'img_%05d'%(int(img_la[7][4:-4])+hh)+'.jpg')
            Apick_path = os.path.join(img_la[3], img_la[5], img_la[6], 'img_%05d'%(int(img_la[7][4:-4])+hh)+'_asp.h5') 

            image_pick = Image.open(Vpick_path).convert('RGB')
            image_pick = self.transform_im(image_pick)
            image_names.append(Vpick_path)
            images.append(image_pick)

            with h5py.File(Apick_path, 'r') as hf:
                audio_features = np.float32(hf['dataset'][:])  # 5,128
            audio_pick = torch.from_numpy(audio_features).float()
            audios.append(audio_pick)
            
        label = np.zeros(28, dtype=np.float32)
        label[inda] = 1
        return img_name, image, audio, inda, label, image_names[0], images[0], audios[0], inda, label,image_names[2], images[2], audios[2], inda, label

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
                for picp in range(1, len(picname)-1, 30):
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


    

    
