import os
import warnings

import face_alignment
import imageio as imageio
import numpy as np
import torch
from scipy.spatial import ConvexHull
from skimage.transform import resize
from skimage.util import img_as_ubyte
from tqdm import tqdm

from demo import load_checkpoints
from demo import make_animation

warnings.filterwarnings("ignore")


def find_best_frame(source, driving, cpu):
    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
    #                                   device= 'cpu' if cpu else 'cuda')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving), total=len(driving), ascii=' >=', desc='Finding best frame'):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num


def deepfake_generation_main(source_image_path, driving_video_path, output_video_path):
    # edit the config
    device = torch.device('cuda:0')
    config_path = 'config/vox-256.yaml'
    checkpoint_path = './checkpoints/vox.pth.tar'
    pixel = 256  # for vox, taichi and mgif, the resolution is 256*256, ted 384*384
    predict_mode = 'relative'  # ['standard', 'relative', 'avd']
    _find_best_frame = True  # when use the relative mode to animate a face


    # parser = ArgumentParser(description='Custom run script for TPSMM')
    # parser.add_argument("--driving", help='Driving performer name', required=True)
    # parser.add_argument("--source", help='Source actor image name', required=True)
    # parser.add_argument("--output", help='Path to output mp4 file', required=True)
    # parser.add_argument("--add_audio", action=BooleanOptionalAction, default=True)
    # parser.add_argument("--output_w_audio", help='Path to output mp4 file with audio', required=False, default='')
    # args = vars(parser.parse_args())

    assert os.path.exists(source_image_path), f'Source image path does not exist. \n {source_image_path}'
    assert os.path.exists(driving_video_path), f'Driving video path does not exist. \n {driving_video_path}'

    # read driving frames and source
    source_image = imageio.imread(source_image_path)
    reader = imageio.get_reader(driving_video_path)
    if len(source_image.shape) < 3:  # gray (one-channel) to rgb (three-channel)
        source_image = np.stack((source_image,) * 3, axis=-1)
    source_image = resize(source_image, (pixel, pixel))[..., :3]

    fps = reader.get_meta_data()['fps']
    print('> Reading files... \t Done.\n> Resizing driving video... ', end='\t')

    driving_video = []
    try:
        for im in reader:
            driving_video.append(resize(im, (pixel, pixel))[..., :3])
    except RuntimeError:
        pass
    reader.close()
    print('Done.')

    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # run model
    # if predict_mode == 'relative' and _find_best_frame:
    i = find_best_frame(source_image, driving_video, device.type == 'cpu')  # cpu
    print("Best frame: " + str(i))
    driving_forward = driving_video[i:]
    driving_backward = driving_video[:(i + 1)][::-1]

    predictions_forward = make_animation(
        source_image,
        driving_forward,
        inpainting,
        kp_detector,
        dense_motion_network,
        avd_network,
        device=device,
        mode=predict_mode,
        desc='Forward'
    )
    predictions_backward = make_animation(
        source_image,
        driving_backward,
        inpainting,
        kp_detector,
        dense_motion_network,
        avd_network,
        device=device,
        mode=predict_mode,
        desc='Backward'
    )
    predictions = predictions_backward[::-1] + predictions_forward[1:]
    # else:
    #     predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network,
    #                                  avd_network, device=device, mode=predict_mode)

    # save output
    imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    print('Saved:\t', output_video_path)

    return output_video_path

