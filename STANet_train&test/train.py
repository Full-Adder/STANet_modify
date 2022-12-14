import os
import torch
import argparse
import numpy as np
import time
import shutil
from loss import KLDLoss
import torch.optim as optim
from STAmodel import STANet
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter
from utils.LoadData import train_data_loader
import cv2
import torch.nn as nn

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    '''parser.add_argument("--img_dir1", type=str, default='/home/omnisky/data/wgt/train/')
    parser.add_argument("--img_dir2", type=str, default='/home/omnisky/data/wgt/test/')
    parser.add_argument("--audio_dir", type=str, default='/home/omnisky/data/wgt/Audio3/')
    parser.add_argument("--swtich_dir1", type=str, default='/home/omnisky/data/wgt/train_switch/')
    parser.add_argument("--swtich_dir2", type=str, default='/home/omnisky/data/wgt/test_switch/')'''
    parser.add_argument("--img_dir1", type=str, default='G:\\Data\\train\\')
    parser.add_argument("--img_dir2", type=str, default='E:\\STAViS-master\\data\\video_frames\\')
    parser.add_argument("--audio_dir1", type=str, default='G:\\Data\\Audio3\\')
    parser.add_argument("--audio_dir2", type=str, default='E:\\STAViS-master\\data\\audio_feature\\')
    parser.add_argument("--swtich_dir1", type=str, default='G:\\Data\\train_switch\\')
    parser.add_argument("--swtich_dir2", type=str, default='')
    parser.add_argument("--groundt_dir1", type=str, default='G:\\Data\\cvprhh\\1\\')
    parser.add_argument("--groundt_dir2", type=str, default='')
    parser.add_argument("--num_classes", type=int, default=28) 
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--input_size", type=int, default=356)
    parser.add_argument("--crop_size", type=int, default=356)
    parser.add_argument("--batch_size", type=int, default=1) 
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
    model = STANet(refine_rate=args.refine_rate, refine_thresh=args.refine_thresh, middir=args.middir, training_epoch=args.epoch)
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
    loss2 = nn.BCEWithLogitsLoss().cuda()
    loss1 = KLDLoss().cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    total_epoch = args.epoch
    global_counter = args.global_counter

    train_loader, val_loader = train_data_loader(args)
    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    model, optimizer = get_model(args)
    audiocls = torch.load('27001.pt')
    audiocls.cuda().eval()
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
            
            img_name1, img1, aud1, inda1, label1, gt1, img_name2, img2, aud2, inda2, label2, gt2, img_name3, img3, aud3, inda3, label3, gt3 = dat
            label1 = label1.cuda(non_blocking=True)
            label2 = label2.cuda(non_blocking=True)
            label3 = label3.cuda(non_blocking=True)   
            img1 = img1.cuda(non_blocking=True)      
            img2 = img2.cuda(non_blocking=True)   
            img3 = img3.cuda(non_blocking=True)   
            aud1 = aud1.cuda(non_blocking=True)      
            aud2 = aud2.cuda(non_blocking=True)   
            aud3 = aud3.cuda(non_blocking=True)   
            gt1 = gt1.cuda(non_blocking=True)      
            gt2 = gt2.cuda(non_blocking=True)   
            gt3 = gt3.cuda(non_blocking=True)  
            with torch.no_grad():
                switch1 = audiocls(aud1, img1)
                switch2 = audiocls(aud2, img2)
                switch3 = audiocls(aud3, img3)
            
            p04,p03,p02,p14,p13,p12,p24,p23,p22 = model(idx, img_name1, img_name2, img_name3, img1, img2, img3, aud1, aud2, aud3, switch1, switch2, switch3, current_epoch, label1, index)
            index += 1

            loss_train= loss2(p04, gt1)+loss2(p14, gt2)+loss2(p24, gt3)+\
                        loss2(p03, gt1)+loss2(p13, gt2)+loss2(p23, gt3)+\
                        loss2(p02, gt1)+loss2(p12, gt2)+loss2(p22, gt3)+\
                        loss1(F.sigmoid(p04), gt1)+loss1(F.sigmoid(p14), gt2)+loss1(F.sigmoid(p24), gt3)+\
                        loss1(F.sigmoid(p03), gt1)+loss1(F.sigmoid(p13), gt2)+loss1(F.sigmoid(p23), gt3)+\
                        loss1(F.sigmoid(p02), gt1)+loss1(F.sigmoid(p12), gt2)+loss1(F.sigmoid(p22), gt3)
                        

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses.update(loss_train.data.item(), img1.size()[0])
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
                        img_name1, img1, aud1, inda1, label1, img_name2, img2, aud2, inda2, label2, img_name3, img3, aud3, inda3, label3 = dat_test
                        label1 = label1.cuda(non_blocking=True)
                        img1 = img1.cuda(non_blocking=True)      
                        img2 = img2.cuda(non_blocking=True)   
                        img3 = img3.cuda(non_blocking=True)   
                        aud1 = aud1.cuda(non_blocking=True)      
                        aud2 = aud2.cuda(non_blocking=True)   
                        aud3 = aud3.cuda(non_blocking=True)   
                        with torch.no_grad():
                            switch1 = audiocls(aud1, img1)
                            switch2 = audiocls(aud2, img2)
                            switch3 = audiocls(aud3, img3)
                        p04,p03,p02,p14,p13,p12,p24,p23,p22 = model(idx, img_name1, img_name2, img_name3, img1, img2, img3, aud1, aud2, aud3, switch1, switch2, switch3, current_epoch, label1, index)
                        batch_num = p12.size()[0]
                        for ii in range(batch_num):
                            file = img_name1[ii].split('\\')[-3]
                            imgp = img_name1[ii].split('\\')[-2]
                            imgn = img_name1[ii].split('\\')[-1]
                            save_path_hh = './runs/pascal2/test/feat/'
                            accu_map_name = os.path.join(save_path_hh, str(save_index), file, imgp, imgn)
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index))):
                                os.mkdir(os.path.join(save_path_hh, str(save_index)))
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index), file)):
                                os.mkdir(os.path.join(save_path_hh, str(save_index), file))
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index), file, imgp)):
                                os.mkdir(os.path.join(save_path_hh, str(save_index), file, imgp))
                            atts = F.sigmoid(p12[ii, 0])
                            att = atts.cpu().data.numpy()
                            att = np.rint(att / (att.max() + 1e-8) * 255)
                            att = np.array(att, np.uint8)
                            att = cv2.resize(att, (356, 356))
                            cv2.imwrite(accu_map_name[:-4]+'.png', att)
                            heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
                            img = cv2.imread(img_name1[ii])
                            img = cv2.resize(img, (356, 356))
                            result = heatmap * 0.3 + img * 0.5
                            cv2.imwrite(accu_map_name, result)

                            '''atts = F.sigmoid(p12[ii][0])
                            att = atts.cpu().data.numpy()
                            att = (att*255.).astype(np.int)/255.
                            att = gaussian_filter(att, sigma=7)
                            att = (att/np.max(att)*255.).astype(np.uint8)
                            att = cv2.resize(att, (356, 356))
                            cv2.imwrite(accu_map_name[:-4]+'.png', att)
                            heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
                            img = cv2.imread(img_name1[ii])
                            img = cv2.resize(img, (356, 356))
                            result = heatmap * 0.3 + img * 0.5
                            cv2.imwrite(accu_map_name, result)'''
                savepath = os.path.join('./runs/pascal2/test/model/', str(save_index)+'.pth')
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
