import os
import random
import shutil

import cv2
import warnings

import gradio as gr
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

    assert os.path.exists(source_image_path), f'Source image path does not exist. \n {source_image_path}'
    assert os.path.exists(driving_video_path), f'Driving video path does not exist. \n {driving_video_path}'

    # read driving frames and source
    source_image = imageio.imread(source_image_path)
    reader = imageio.get_reader(driving_video_path)
    if len(source_image.shape) < 3:  # gray (one-channel) to rgb (three-channel)
        source_image = np.stack((source_image,) * 3, axis=-1)

    # TODO: source image crop to face
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

    # save output
    imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    print('Saved:\t', output_video_path)

    return os.path.exists(output_video_path)


def crop_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('./checkpoints/haarcascade_frontalface_alt2.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = img[y:y + h, x:x + w]
        return face


def process_files(video, image):
    # Process the video and image files here
    # After processing, you could upload the video to a hosting platform and get a URL
    # This is a placeholder URL, replace with your actual video URL

    rand_num = random.randint(9999, 100000)
    if not os.path.basename(video).endswith('.mp4'):
        return ''
    driving_video_save_path = f'./media/driving/driving_{rand_num}.mp4'
    source_image_save_path = f'./media/source/source_{rand_num}.jpg'
    output_video_save_path = f'./media/output/output_{rand_num}.mp4'

    shutil.copy(video, driving_video_save_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(source_image_save_path, image)

    if deepfake_generation_main(source_image_save_path, driving_video_save_path, output_video_save_path):
        return output_video_save_path
    return ''


if __name__ == '__main__':
    # Define the interface
    iface = gr.Interface(
        fn=process_files,
        inputs=[
            gr.components.Video(label="Input Video", autoplay=True),
            gr.components.Image(label="Input Image")
        ],
        # examples=[["./media/driving/driving.mp4"], ["./media/source/source.png"]],
        outputs=gr.components.Video(autoplay=True),
        title="Simple Deepfake Generation",
        description="""
                       Upload a video and an image, and get a animated deepfake.
                    """,
        cache_examples=True
    )

    # Launch the app
    iface.launch()
    # iface.launch(share=True)
