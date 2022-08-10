import torch
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
from scipy.io import wavfile
import os
import h5py
from PIL import Image
from scipy import signal
import cv2
import random
import soundfile as sf
import resampy
import numpy as np
import json
import argparse
import math
import os
from torchvision import datasets, transforms
'''
video_path = "E:\\STAViS-master\\data\\video_frames\\"
audio_dir = "E:\\STAViS-master\\data\\video_audio\\"
feature_path = "E:\\STAViS-master\\data\\audio_feature\\"
'''
video_path = "G:\\Data\\Img2\\" 
audio_dir = "G:\\Data\\Audio2\\" 
feature_path = "G:\\Data\\Audio4\\"
ori_name = os.listdir(video_path)
for file in range(0, len(ori_name)):
    print(ori_name[file])
    if not os.path.exists(os.path.join(feature_path, ori_name[file])):
        os.makedirs(os.path.join(feature_path, ori_name[file]))
    ficpath = os.path.join(video_path, ori_name[file])
    ficname = os.listdir(ficpath)
    for fs in range(0, len(ficname)):
        save_path = os.path.join(feature_path, ori_name[file], ficname[fs])
        print(ficname[fs])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        picpath = os.path.join(ficpath, ficname[fs])
        picname = os.listdir(picpath)
        newlist = []
        for names in picname:
            if names.endswith(".jpg"):
                newlist.append(names)

        audio_path = os.path.join(
            audio_dir, ori_name[file], ficname[fs], ficname[fs] + '.wav')
        
        if os.path.exists(audio_path):

            samples, samplerate = sf.read(audio_path)

            if len(samples.shape) > 1:
                samples = np.mean(samples, axis=1)

            SAMPLE_RATE = 16000
            if samplerate != SAMPLE_RATE:
                samples = resampy.resample(samples, samplerate, SAMPLE_RATE)  # 采样速率转换44100->16000
            num = samples.size // SAMPLE_RATE
            nuf = len(newlist)
            for picp in range(0, num):
                s = picp*16000
                e = (picp+1)*16000
                resamples = samples[s:e]
                resamples[resamples > 1.] = 1.
                resamples[resamples < -1.] = -1.
                _, _, spectrogram = signal.spectrogram(resamples, SAMPLE_RATE, nperseg=160, noverlap=80)
                spectrogram = np.log(spectrogram + 1e-7)
                mean = np.mean(spectrogram)
                std = np.std(spectrogram)
                audio_output = np.divide(spectrogram-mean, std+1e-9)  # 257，61
                print(newlist[picp])
                with h5py.File(save_path+'\\audio_'+ '%05d' % (picp) +'_asp.h5', 'w') as hf:
                    hf.create_dataset("dataset",  data=audio_output)

