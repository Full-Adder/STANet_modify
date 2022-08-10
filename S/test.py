import os
import torch
import argparse
import numpy as np
import shutil
import torch.optim as optim
from Smodel import SNetModel
import torch.nn.functional as F
from utils.LoadData import test_data_loader
import cv2
import os 



ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    '''parser.add_argument("--img_dir1", type=str, default='/home/omnisky/data/wgt/train/')
    parser.add_argument("--img_dir2", type=str, default='/home/omnisky/data/wgt/test/')
    parser.add_argument("--audio_dir", type=str, default='/home/omnisky/data/wgt/Audio3/')
    parser.add_argument("--swtich_dir1", type=str, default='/home/omnisky/data/wgt/train_switch/')
    parser.add_argument("--swtich_dir2", type=str, default='/home/omnisky/data/wgt/train_switch/')'''
    parser.add_argument("--img_dir1", type=str, default='G:\\Data\\train\\')
    parser.add_argument("--img_dir2", type=str, default='G:\\Data\\test\\')
    parser.add_argument("--audio_dir", type=str, default='G:\\Data\\Audio3\\')
    parser.add_argument("--swtich_dir1", type=str, default='G:\\Data\\train_switch\\')
    parser.add_argument("--swtich_dir2", type=str, default='G:\\Data\\train_switch\\')
    parser.add_argument("--num_classes", type=int, default=28) 
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--decay_points", type=str, default='5,10')
    parser.add_argument("--snapshot_dir", type=str, default='runs/')
    parser.add_argument("--middir", type=str, default='runs/')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=5)

    return parser.parse_args()


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))


def get_model(args):
    model = SNetModel(middir=args.middir)
    device = torch.device(0)	
    model = torch.nn.DataParallel(model).cuda()
    model.to(device)
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    return  model, optimizer


# -------------------以上和train保持一致---------------------------

def test(args):
    model, optimizer = get_model(args)
    val_loader = test_data_loader(args)         # 获取测试集
    idx_save = 0
    for idx_test, dat_test in enumerate(val_loader):    # idx_test：数据编号， dat_test：（图像名，图像，编号，标签）
        model.eval()                            # 评估模式
        img_name1, img1, inda1, label1 = dat_test
        label1 = label1.cuda(non_blocking=True)
        img1 = img1.cuda(non_blocking=True)       
                    
        x11, x22, map1, map2 = model(1, img_name1, img1, 1, label1, 1)  # 前向传播
        
        h_x = F.softmax(x11, dim=1).data.squeeze()                      # softmax 转化为概率
        probs, indxofp = h_x.sort(1, True)  # 1行排序                     #降序
        probs = probs[:,0]                                              # 排序后最大数值
        indxofp = indxofp[:,0]                                          # 排序后最大值索引
        
        batch_num = img1.size()[0]  # 3
        ind = torch.nonzero(label1)  # [10, 28] -> 非0元素的行列索引
        for i in range(batch_num):  # 非0元素的个数
            batch_index, la = ind[i]  # 帧索引，类别索引
            if la == indxofp[i]:                                        # 如果标签中非0索引==最大结果的索引
                file = img_name1[i].split('/')[-3]                      # 下面与train.py 同理
                imgp = img_name1[i].split('/')[-2]
                imgn = img_name1[i].split('/')[-1]   
                featpath = './runs/pascal2/train/all/'
                accu_map_name = os.path.join(featpath, file, imgp, imgn)
                if not os.path.exists(os.path.join(featpath, file)):
                    os.mkdir(os.path.join(featpath, file))
                if not os.path.exists(os.path.join(featpath, file, imgp)):
                    os.mkdir(os.path.join(featpath, file, imgp))
                atts = (map1[i]+map2[i])/2
                atts[atts < 0] = 0

                att = atts[indxofp[i]].cpu().data.numpy()
                att = np.rint(att / (att.max() + 1e-8) * 255)
                att = np.array(att, np.uint8)
                att = cv2.resize(att, (220, 220))
                cv2.imwrite(accu_map_name[:-4]+'.png', att)
                idx_save = idx_save+1
                if idx_save % 100 == 0:
                    heatmap = cv2.applyColorMap(
                        att, cv2.COLORMAP_JET)
                    img = cv2.imread(img_name1[i])
                    img = cv2.resize(img, (220, 220))
                    result = heatmap * 0.3 + img * 0.5
                    cv2.imwrite(accu_map_name, result)

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    save_index = 0
    test(args)
