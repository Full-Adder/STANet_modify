import os
import torch
import argparse
import numpy as np
import time
import shutil
import torch.optim as optim
from modelCGCN import conservativeCGCN
import torch.nn.functional as F
from utils import AverageMeter
from utils.LoadData import train_data_loader
import cv2

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)

import os
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
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--refine_rate", type=float, default=0.6)
    parser.add_argument("--refine_thresh", type=float, default=0.8)
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
    model = conservativeCGCN(pretrained=True, refine_rate=args.refine_rate, refine_thresh=args.refine_thresh, middir=args.middir, training_epoch=args.epoch)
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


def train(args, save_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    total_epoch = args.epoch
    global_counter = args.global_counter

    train_loader, val_loader = train_data_loader(args)
    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    model, optimizer = get_model(args)
    
    model.train()
    print(model)
    end = time.time()
    current_epoch = 0

    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        batch_time.reset()
        steps_per_epoch = len(train_loader)

        index = 0  
        for idx, dat in enumerate(train_loader):
            
            img_name0_0, img0_0, img_name0_1, img0_1, img_name0_2, img0_2, \
            img_name1_0, img1_0, img_name1_1, img1_1, img_name1_2, img1_2, \
            img_name2_0, img2_0, img_name2_1, img2_1, img_name2_2, img2_2, \
            inda, label = dat
            label = label.cuda(non_blocking=True) 
            img0_0 = img0_0.cuda(non_blocking=True)      
            img0_1 = img0_1.cuda(non_blocking=True)   
            img0_2 = img0_2.cuda(non_blocking=True)   
            img1_0 = img1_0.cuda(non_blocking=True)      
            img1_1 = img1_1.cuda(non_blocking=True)   
            img1_2 = img1_2.cuda(non_blocking=True)   
            img2_0 = img2_0.cuda(non_blocking=True)      
            img2_1 = img2_1.cuda(non_blocking=True)   
            img2_2 = img2_2.cuda(non_blocking=True)   
            x0_0ss,x0_1ss, x0_1sss,x0_2ss, x1_0ss,x1_1ss,x1_1sss,x1_2ss, x2_0ss,x2_1ss, x2_1sss,x2_2ss, map1, map_all = model(idx, img_name0_0, img_name0_1, img_name0_2, img_name1_0, img_name1_1, img_name1_2, img_name2_0, img_name2_1, img_name2_2, 
            img0_0, img0_1, img0_2, img1_0, img1_1, img1_2, img2_0, img2_1, img2_2, current_epoch, label, index)
            index += 1

            loss_train = 0.4 * (F.multilabel_soft_margin_loss(x0_0ss, label) + F.multilabel_soft_margin_loss(x0_1ss, label) + F.multilabel_soft_margin_loss(x0_2ss, label)\
                    + F.multilabel_soft_margin_loss(x1_0ss, label) + F.multilabel_soft_margin_loss(x1_1ss, label) + F.multilabel_soft_margin_loss(x1_2ss, label)\
                    + F.multilabel_soft_margin_loss(x2_0ss, label) + F.multilabel_soft_margin_loss(x2_1ss, label) + F.multilabel_soft_margin_loss(x2_2ss, label)) \
                    + (F.multilabel_soft_margin_loss(x0_1sss, label)+F.multilabel_soft_margin_loss(x1_1sss, label) + F.multilabel_soft_margin_loss(x2_1sss, label))

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses.update(loss_train.data.item(), img0_0.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()
            
            global_counter += 1

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses))
            if global_counter % 500 == 0:
                save_index = save_index + 1
                with torch.no_grad():
                    for idx_test, dat_test in enumerate(val_loader):
                        model.eval()
                        img_name0_0, img0_0, img_name0_1, img0_1, img_name0_2, img0_2, \
                        img_name1_0, img1_0, img_name1_1, img1_1, img_name1_2, img1_2, \
                        img_name2_0, img2_0, img_name2_1, img2_1, img_name2_2, img2_2, \
                        inda, label = dat_test
                        label = label.cuda(non_blocking=True) 
                        img0_0 = img0_0.cuda(non_blocking=True)      
                        img0_1 = img0_1.cuda(non_blocking=True)   
                        img0_2 = img0_2.cuda(non_blocking=True)   
                        img1_0 = img1_0.cuda(non_blocking=True)      
                        img1_1 = img1_1.cuda(non_blocking=True)   
                        img1_2 = img1_2.cuda(non_blocking=True)   
                        img2_0 = img2_0.cuda(non_blocking=True)      
                        img2_1 = img2_1.cuda(non_blocking=True)   
                        img2_2 = img2_2.cuda(non_blocking=True)  
                        
                        x0_0ss,x0_1ss, x0_1sss,x0_2ss, x1_0ss,x1_1ss,x1_1sss,x1_2ss, x2_0ss,x2_1ss, x2_1sss,x2_2ss, map1, map_all = model(idx, img_name0_0, img_name0_1, img_name0_2, img_name1_0, img_name1_1, img_name1_2, img_name2_0, img_name2_1, img_name2_2, 
                        img0_0, img0_1, img0_2, img1_0, img1_1, img1_2, img2_0, img2_1, img2_2, current_epoch, label, index)
                        batch_num = img0_0.size()[0]
                        for ii in range(batch_num):
                            ind = torch.nonzero(label)  # [10, 28] -> 非0元素的行列索引
                            batch_index, la = ind[ii]  # 帧索引，类别索引
                            file = img_name0_1[ii].split('/')[-3]
                            imgp = img_name0_1[ii].split('/')[-2]
                            imgn = img_name0_1[ii].split('/')[-1]
                            save_path_hh = './runs/feat/'
                            accu_map_name = os.path.join(save_path_hh, str(save_index), file, imgp, imgn)
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index))):
                                os.mkdir(os.path.join(save_path_hh, str(save_index)))
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index), file)):
                                os.mkdir(os.path.join(save_path_hh, str(save_index), file))
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index), file, imgp)):
                                os.mkdir(os.path.join(save_path_hh, str(save_index), file, imgp))
                            atts = (map1[ii] + map_all[ii]) / 2
                            atts[atts < 0] = 0
                            att = atts[la].cpu().data.numpy()
                            att = np.rint(att / (att.max() + 1e-8) * 255)
                            att = np.array(att, np.uint8)
                            att = cv2.resize(att, (220, 220))
                            cv2.imwrite(accu_map_name[:-4]+'_c.jpg', att)
                            heatmap = cv2.applyColorMap(
                                att, cv2.COLORMAP_JET)
                            img = cv2.imread(img_name0_1[ii])
                            img = cv2.resize(img, (220, 220))
                            result = heatmap * 0.3 + img * 0.5
                            cv2.imwrite(accu_map_name, result)
                savepath = os.path.join('./runs/model/', str(save_index)+'.pth')
                torch.save(model.state_dict(), savepath)
                model.train()
        if current_epoch == args.epoch-1:
            save_checkpoint(args,
                        {
                            'epoch': current_epoch,
                            'global_counter': global_counter,
                            'state_dict':model.state_dict(),
                            'optimizer':optimizer.state_dict()
                        }, is_best=False,
                        filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    save_index = 0
    train(args, save_index)
