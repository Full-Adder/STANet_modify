from .transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


def train_data_loader(args, test_path=False, segmentation=False):
    mean_vals = [0.485, 0.456, 0.406]           # 数据均值
    std_vals = [0.229, 0.224, 0.225]            # 数据标准差
    input_size = int(args.input_size)           # 输入大小
    crop_size = int(args.crop_size)             # 裁剪大小
    tsfm_train = transforms.Compose([transforms.Resize(input_size),         # 修改分辨率
                                     transforms.RandomCrop(crop_size),      # 随机裁剪
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # 修改亮度、名度等
                                     transforms.ToTensor(), # 转变为tensor，并/255
                                     transforms.Normalize(mean_vals, std_vals),   # 归一化
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize(input_size),                 # 同理
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])

    img_train = VOCDatasetT(root_dir=args.img_dir1, num_classes=args.num_classes, transform=tsfm_train, test=False)
    img_test = VOCDatasetE(root_dir=args.img_dir2, num_classes=args.num_classes, transform=tsfm_test, test=True)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=3, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def test_data_loader(args, test_path=False, segmentation=False):        # 同理
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    input_size = int(args.input_size)
    tsfm_test = transforms.Compose([transforms.Resize(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])

    img_test = VOCDatasetE(root_dir=args.img_dir2, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader与val_loader 在数据上是一致的，但batch_size大小不同

    return val_loader


class VOCDatasetT(Dataset):  # 数据集类
    def __init__(self, root_dir, num_classes=28, transform=None, test=False):
        self.root_dir = root_dir  # 数据存放路径
        self.testing = test  # 是否是训练集
        self.transform = transform  # 数据转换到tensor等一系列数据预处理
        self.num_classes = num_classes  # 类别数量
        self.image_list, self.name_list, self.figs_list, self.seqs_list = self.read_labeled_image_list_train(
            self.root_dir)
        self.image_num = 2  # 没用到。。
        self.thres = 1000  # 没用到。。

    def __len__(self):  # 数据集大小
        return len(self.image_list)

    def __getitem__(self, idx):  # 获得第idx号数据
        pathimla = self.image_list[idx]
        img_la = pathimla.split('+')  # [pic_absdir,0,root_dir,fic_dir,..,pic.jpg]

        img_name = img_la[0]  # 图片绝对地址
        image = Image.open(img_name).convert('RGB')  # 打开图片，并转化为RGB格式
        image = self.transform(image)  # 将图片转化为tensor

        inda = int(img_la[1])  # 获得图像标签
        label = np.zeros(28, dtype=np.float32)
        label[inda] = 1  # one-hot编码
        return img_name, image, inda, label  # 返回 图片地址 图片tensor 标签 onehot标签

    def read_labeled_image_list_train(self, data_dir):
        '''
        data_dir(AVE):
        ----ficpath(ori_name[i]): dir_name = 标签类别[0,1,2,...,27]
            ----picpath1(ficname[i]):
                ----picname1(picpath[i]).jpg
                ----picname2.jpg
                ...
            ----picpath2:
            ...
        '''
        path_list = []  # 双重地址（pic_absdir+every_dir1+every_dir2+ ··· +pic_name）
        name_list = []  # 名字
        figs_list = []  # 图像
        seqs_list = []  # 文件夹
        ori_name = os.listdir(data_dir)  # 根目录下的所有目录
        ori_name.sort()  # 排序
        for file in range(0, len(ori_name)):  # 进入第一个目录 （或者进入所有目录）
            print(file)
            ficpath = os.path.join(data_dir, ori_name[file])  # 拼接进入的目录地址（AVE数据集）
            ficname = os.listdir(ficpath)  # 获得拼接目录下的所有文件夹
            ficname.sort()  # 排序
            num_list = []
            fig_list = []
            seq_list = []
            for fs in range(0, len(ficname)):  # 遍历文件夹
                picpath = os.path.join(ficpath, ficname[fs])  # 拼接完整路径
                picname = os.listdir(picpath)  # 获取文件夹中所有图片的路径
                picname.sort()  # 排序
                if len(picname) < 3:  # 忽略不包含图片的文件夹
                    continue
                for picp in range(0, len(picname)):  # 遍历图片
                    if picname[picp].endswith('.jpg'):  # 如果是图片
                        pv1 = os.path.join(data_dir, ori_name[file], ficname[fs], picname[picp])  # 拼接完整路径
                        path_list.append(
                            pv1 + '+' + str(file) + '+' + data_dir + '+' + ori_name[file] + '+' + ficname[fs] + '+' +
                            picname[picp][:-4] + '.jpg')  # 拼接的双重路径
                num_list.append([ficname[fs], len(picname)])  # 记录文件夹名和其中图片的数量[[fn1,len1],[fn1,len2],...]
                fig_list.append(picname)  # 记录图片名[[pic1,pic2,pic3,..][pic1,pic2,...][...]]
                seq_list.append(ficname[fs])  # 记录总文件夹名[fic1,fic2,...]
            name_list.append(num_list)  # 汇总文件夹+数量 [[[fn1,len1],[fn1,len2],...]]
            figs_list.append(fig_list)  # 汇总所有图片 [[[pic1,pic2,pic3,..][pic1,pic2,...][...]]]
            seqs_list.append(seq_list)  # 汇总所有文件夹[[fic1,fic2,...]]
        return path_list, name_list, figs_list, seqs_list


class VOCDatasetE(Dataset):  # 与上面一致，只是函数名不同
    def __init__(self, root_dir, num_classes=28, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.name_list, self.figs_list, self.seqs_list = self.read_labeled_image_list_test(
            self.root_dir)
        self.image_num = 2
        self.thres = 1000

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        pathimla = self.image_list[idx]
        img_la = pathimla.split('+')

        img_name = img_la[0]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        inda = int(img_la[1])
        label = np.zeros(28, dtype=np.float32)
        label[inda] = 1
        return img_name, image, inda, label

    def read_labeled_image_list_test(self, data_dir):
        path_list = []
        name_list = []
        figs_list = []
        seqs_list = []
        ori_name = os.listdir(data_dir)
        ori_name.sort()
        for file in range(0, len(ori_name)):
            # for file in range(0, 1):  # len(ori_name)):
            print(file)
            ficpath = os.path.join(data_dir, ori_name[file])
            ficname = os.listdir(ficpath)
            ficname.sort()
            num_list = []
            fig_list = []
            seq_list = []
            for fs in range(0, len(ficname)):
                picpath = os.path.join(ficpath, ficname[fs])
                picname = os.listdir(picpath)
                picname.sort()
                if len(picname) < 3:
                    continue
                for picp in range(1, len(picname)):
                    if picname[picp].endswith('.jpg'):
                        pv1 = os.path.join(data_dir, ori_name[file], ficname[fs], picname[picp])
                        path_list.append(
                            pv1 + '+' + str(file) + '+' + data_dir + '+' + ori_name[file] + '+' + ficname[fs] + '+' +
                            picname[picp][:-4] + '.jpg')
                num_list.append([ficname[fs], len(picname)])
                fig_list.append(picname)
                seq_list.append(ficname[fs])
            name_list.append(num_list)
            figs_list.append(fig_list)
            seqs_list.append(seq_list)
        return path_list, name_list, figs_list, seqs_list
