import os
import torch
import argparse     # 命令行操作
import numpy as np
import time
import shutil       # 文件操作
import my_optim     # 作者的优化器
import torch.optim as optim
from Smodel import SNetModel
import torch.nn.functional as F
from utils import AverageMeter
from utils.LoadData import train_data_loader
import cv2

ROOT_DIR = '/'.join(os.getcwd().split('\\')[:-1])
# 项目原代码 ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1]) 有问题！
print('Project Root Dir:', ROOT_DIR)
# Project Root Dir: D:/WorkPlace/Python/STANet-main
# 设置文件root路径，并打印。

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置使用GPU：0作为训练设备。


def get_arguments():    # 获取命令行参数的函数
    parser = argparse.ArgumentParser()  # 设置命令行参数解析器
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)   # 添加参数解析项 --root_dir 项目根目录
    '''parser.add_argument("--img_dir1", type=str, default='/home/omnisky/data/wgt/train/')
    parser.add_argument("--img_dir2", type=str, default='/home/omnisky/data/wgt/test/')
    parser.add_argument("--audio_dir", type=str, default='/home/omnisky/data/wgt/Audio3/')
    parser.add_argument("--swtich_dir1", type=str, default='/home/omnisky/data/wgt/train_switch/')
    parser.add_argument("--swtich_dir2", type=str, default='/home/omnisky/data/wgt/train_switch/')'''
    parser.add_argument("--img_dir1", type=str, default='G:\\Data\\train\\')    # img_dir1 是训练集的地址
    parser.add_argument("--img_dir2", type=str, default='G:\\Data\\test\\')     # img_dir2 是测试集的地址
    # parser.add_argument("--audio_dir", type=str, default='G:\\Data\\Audio3\\')  # audio_dir 是音频文件地址
    # parser.add_argument("--swtich_dir1", type=str, default='G:\\Data\\train_switch\\')  # 转换后训练集
    # parser.add_argument("--swtich_dir2", type=str, default='G:\\Data\\train_switch\\')  # 转换后测试集
    parser.add_argument("--num_classes", type=int, default=28)                  # 数据集中分类的种类
    parser.add_argument("--lr", type=float, default=0.0001)                     # lr(learning rate) 优化器的学习率
    parser.add_argument("--weight_decay", type=float, default=0.0005)           # 权重衰减
    parser.add_argument("--input_size", type=int, default=300)                  # 输入大小
    parser.add_argument("--crop_size", type=int, default=256)                   # 图片的裁剪大小
    parser.add_argument("--batch_size", type=int, default=8)                    # 批大小
    parser.add_argument("--decay_points", type=str, default='5,10')             # 衰变点
    parser.add_argument("--snapshot_dir", type=str, default='runs/')            # 快照存储地址
    parser.add_argument("--middir", type=str, default='runs/')                  # 没看怎么用，可能是存储中间过程 的地址
    parser.add_argument("--num_workers", type=int, default=0)                   # DataLoad的工作进程 0=只有主进程加载
    parser.add_argument("--epoch", type=int, default=15)                        # 训练轮数
    parser.add_argument("--global_counter", type=int, default=0)                # 全局计数器
    parser.add_argument("--disp_interval", type=int, default=5)                 # 展示间隔

    return parser.parse_args()


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):   # 函数：保存检查点（命令行参数，状态，是否是最优，文件名）
    savepath = os.path.join(args.snapshot_dir, filename)                    # 从命令行参数中获取快照存储路径与文件名组成存储路径
    torch.save(state, savepath)                                             # torch.save 方法及将 state 保存在 savepath
    if is_best:                                                             # 如果是最好的
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))    # 文件命名best复制到快照文件夹


def get_model(args):                                # 获取 model
    model = SNetModel(middir=args.middir)           # 获得S
    device = torch.device(0)                        # GPU
    model = torch.nn.DataParallel(model).cuda()     # 多卡数据并行训练 torch.nn.DataParallel(model,device = all)
    model.to(device)                                # 加载到GPU:0
    param_groups = model.module.get_parameter_groups()  # 获取 1作者定义的网络的权重 2作者定义的网罗的偏置 3resnext的权重 4resnext的偏置
    optimizer = optim.SGD([                                 # 随机梯度下降
        {'params': param_groups[0], 'lr': args.lr},         # 对于不同类型参数用不同的学习率
        {'params': param_groups[1], 'lr': 2 * args.lr},
        {'params': param_groups[2], 'lr': 10 * args.lr},
        {'params': param_groups[3], 'lr': 20 * args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    return model, optimizer                             # 返回模型和优化器


def train(args, save_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    '''AverageMeter():
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    '''

    total_epoch = args.epoch                # 总轮数
    global_counter = args.global_counter    # 全局计数器
    current_epoch = 0                       # 当前轮数

    train_loader, val_loader = train_data_loader(args)  # data_load--> 训练集 验证集
    max_step = total_epoch * len(train_loader)          # 最大更新次数
    args.max_step = max_step
    print('Max step:', max_step)

    model, optimizer = get_model(args)                  # model 和 优化器

    model.train()                                       # train!
    print(model)
    end = time.time()

    while current_epoch < total_epoch:                  # 当前轮数小于总共需要的轮数
        model.train()                                   # train
        losses.reset()                                  # reset()置零
        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch) # 优化器在衰减点的轮数改变lr
        steps_per_epoch = len(train_loader)  # 每轮的更新次数

        index = 0
        for idx, dat in enumerate(train_loader):    # 从train_loader中获取数据

            img_name1, img1, inda1, label1 = dat
            label1 = label1.cuda(non_blocking=True)

            x11, x22, map1, map2 = model(idx, img_name1, img1, current_epoch, label1, index)   # model前向传播
            index += 1                                                                         # x2比x1多了attention
            loss_train = F.multilabel_soft_margin_loss(x11, label1) + F.multilabel_soft_margin_loss(x22, label1)
            # loss

            optimizer.zero_grad()   # 梯度置0
            loss_train.backward()   # 反向传播
            optimizer.step()        # 优化一次

            losses.update(loss_train.data.item(), img1.size()[0])   # 计算平均的loss
            batch_time.update(time.time() - end)                    # 计算每个batch的平均时间
            end = time.time()                                       # 更新结束时间

            global_counter += 1

            if global_counter % args.disp_interval == 0:           # 该打印输出了
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, global_counter % len(train_loader), len(train_loader),
                    optimizer.param_groups[0]['lr'], loss=losses))
            if global_counter % 500 == 0:                           # 500个轮次验证一次
                with torch.no_grad():                               # 不计梯度加快验证
                    save_index = save_index + 1
                    for idx_test, dat_test in enumerate(val_loader):    # 取验证集数据
                        model.eval()                                    # 评估模式
                        img_name1, img1, inda1, label1 = dat_test
                        label1 = label1.cuda(non_blocking=True)

                        x11, x22, map1, map2 = model(
                            idx_test, img_name1, img1, current_epoch, label1, index) # 前向传播
                        batch_num = img1.size()[0]  # 3
                        ind = torch.nonzero(label1)  # [10, 28] -> 非0元素的行列索引
                        for i in range(ind.shape[0]):  # 非0元素的个数
                            batch_index, la = ind[i]  # 帧索引，类别索引
                            file = img_name1[i].split('/')[-3]  # 文件夹名
                            imgp = img_name1[i].split('/')[-2]  # 图片文件夹名
                            imgn = img_name1[i].split('/')[-1]  # 图片名
                            save_path_hh = './runs/feat/'   # 保存地址
                            accu_map_name = os.path.join(save_path_hh, str(save_index), file, imgp, imgn)
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index))):     # 创建文件夹
                                os.mkdir(os.path.join(save_path_hh, str(save_index)))
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index), file)):
                                os.mkdir(os.path.join(save_path_hh, str(save_index), file))
                            if not os.path.exists(os.path.join(save_path_hh, str(save_index), file, imgp)):
                                os.mkdir(os.path.join(save_path_hh, str(save_index), file, imgp))
                            atts = (map1[i] + map2[i]) / 2          # 计算两幅图的平均值
                            atts[atts < 0] = 0                      # 消除负值
                            att = atts[la].cpu().data.numpy()       # 转为numpy数组
                            att = np.rint(att / (att.max() + 1e-8) * 255)   # 归一化到0-255
                            att = np.array(att, np.uint8)
                            att = cv2.resize(att, (220, 220))       # 修改分辨率
                            cv2.imwrite(accu_map_name[:-4] + '.png', att)   # 保存图片
                            heatmap = cv2.applyColorMap(
                                att, cv2.COLORMAP_JET)              # 制作att伪彩色图像
                            img = cv2.imread(img_name1[i])
                            img = cv2.resize(img, (220, 220))
                            result = heatmap * 0.3 + img * 0.5      # 将原图和伪彩色图像重叠起来
                            cv2.imwrite(accu_map_name, result)      # 保存图像
                savepath = os.path.join('./runs/model/', str(save_index) + '.pth')
                torch.save(model.state_dict(), savepath)            # 保存现在的权重
                model.train()                                       # 改回训练模式
        if current_epoch == args.epoch - 1:                         # 如果现在的轮数=最终的轮数-1
            save_checkpoint(args,                                   # 保存检查点
                            {
                                'epoch': current_epoch,             # 当前轮数
                                'global_counter': global_counter,   # 优化器次数
                                'state_dict': model.state_dict(),   # 模型参数
                                'optimizer': optimizer.state_dict() # 优化器参数
                            }, is_best=False,
                            filename='%s_epoch_%d.pth' % (args.dataset, current_epoch))
        current_epoch += 1


if __name__ == '__main__':
    args = get_arguments()                          # 获得命令行参数
    print('Running parameters:\n', args)            # 打印
    if not os.path.exists(args.snapshot_dir):       # 确保快照文件夹存在
        os.makedirs(args.snapshot_dir)
    save_index = 0                                  # 已经保存的次数
    train(args, save_index)
