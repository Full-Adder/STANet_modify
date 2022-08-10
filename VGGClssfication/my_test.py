import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from scipy import signal
import random
import soundfile as sf
import resampy
import numpy as np
import json
import argparse
import math
import csv
from model import AVENet

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
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--batch_size', 
        default=32, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=309,
        type=int,
        help=
        'Number of classes')
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


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    args = get_arguments()
    model= AVENet(args) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda()

    classes = []
    with open('data/stat.csv') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
    classes = sorted(classes)
    
    # load pretrained models
    checkpoint = torch.load(args.summaries)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print('load pretrained model.')
    model.eval()
    
    # audio_path = 'FwVYUHKoLtQ_000034.wav'
    audio_path = 'G:/AVE-ECCV18-master/AVE_Dataset/audio2/Goat/-bTFWFcUjNA/-bTFWFcUjNA.wav'
    samples, samplerate = sf.read(audio_path)

    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)

    SAMPLE_RATE = 16000
    if samplerate != SAMPLE_RATE:
        samples = resampy.resample(samples, samplerate, SAMPLE_RATE)  # 采样速率转换44100->16000
    T = 300
    long = 8000
    L = samples.shape[0]
    # IoU = math.floor(long-(L-long-(long/500))/(T-1))
    IoU = math.ceil(long-(L-long)/(T-1))
    spectrogramall = np.zeros([T, 1, 257, 48])
    for i in range(T):
        s = i*(long-IoU)
        e = i*(long-IoU)+long
        resamples = samples[s:e]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        _, _, spectrogram = signal.spectrogram(resamples, SAMPLE_RATE, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram-mean, std+1e-9)  # 257，61
        spectrogramall[i, 0, :, :] = spectrogram

    spec = Variable(torch.from_numpy(spectrogramall)).cuda()
    aud_o = model(spec.float())

    prediction = nn.Softmax(dim=1)(aud_o)
    _,pred = torch.max(prediction.cpu().data, 1)
    for m in range(T):
       print(classes[int(pred[m].numpy())])
    print()



if __name__ == "__main__":
    main()