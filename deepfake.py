import torch
import imageio
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings
import os

warnings.filterwarnings("ignore")


# edit the config
device = torch.device('cuda:0')
dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
config_path = 'config/vox-256.yaml'
checkpoint_path = 'checkpoints/vox.pth.tar'
predict_mode = 'relative'  # ['standard', 'relative', 'avd']
find_best_frame = True  # when use the relative mode, use 'find_best_frame=True' can get better quality result
pixel = 256  # for vox, taichi and mgif, the resolution is 256*256


def generate(source_image_path, driving_video_path, output_video_path):
    pass


if __name__ == '__main__':
    source_image_path = '/data/VideoRecordings/source/'
    driving_video_path = '/data/VideoRecordings/driving/'
    output_video_path = '/data/VideoRecordings/output/'