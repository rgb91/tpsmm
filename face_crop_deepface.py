"""
Use conda environment: deepface0
"""
from pathlib import Path
import os
import cv2
from deepface import DeepFace


def crop_face(video_path: str, out_dir: str, face_resolution=512, padding=50):
    video_name = os.path.basename(video_path).split('.')[0]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    count, err_count, hop = 0, 0, 10
    x_min, y_min = 999999, 999999
    w_max, h_max = -1, -1
    x_max, y_max = -1, -1
    first_frame_with_face = -1

    # get x, y, w, h for sampled frames
    print('Calculating Face Area.')
    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    while success:
        if count % hop == 0:
            # list of ('face', 'facial_area', 'confidence')
            try:
                face_obj = DeepFace.extract_faces(img_path=image, detector_backend='dlib', enforce_detection=True)
                if len(face_obj) < 1:
                    continue
                if first_frame_with_face < 0:
                    first_frame_with_face = count
                face_area = face_obj[0]['facial_area']
                x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']

                x_min, y_min = min(x_min, x), min(y_min, y)
                w_max, h_max = max(w_max, w), max(h_max, h)
                x_max, y_max = max(x_max, x), max(y_max, y)
            except Exception as e:
                err_count += 1
        success, image = video_cap.read()
        count += 1
    video_cap.release()
    print(f'Face Detection Error count = {err_count}.')

    # crop each frame based on the calculated face area above
    print('Cropping Faces.')
    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    count = 0
    err_count = 0
    h_w_max = max(h_max, w_max)
    while success:
        if count < first_frame_with_face:
            count += 1
            continue
        # TODO: add padding check boundary
        crop_image = image[y_min - padding:y_max + h_w_max + padding, x_min - padding:x_max + h_w_max + padding]
        crop_image = cv2.resize(crop_image, (face_resolution, face_resolution))
        frame_path = os.path.join(out_dir, f'{count:05d}.jpg')
        cv2.imwrite(frame_path, crop_image)

        success, image = video_cap.read()
        count += 1
    video_cap.release()
    # crop_image = cv2.resize(face_obj[0]['face'], (224, 224))


def save_frames(video_path: str, out_dir: str):
    video_name = os.path.basename(video_path).split('.')[0]
    print(video_path, out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    count, err_count = 0, 0
    print(success)
    while success:
        frame_path = os.path.join(out_dir, f'{count:05d}.jpg')
        cv2.imwrite(frame_path, image)
        success, image = video_cap.read()
        count += 1
    video_cap.release()


def check_face_detection(image_path, out_path):
    face_obj = DeepFace.detectFace(img_path=image_path,
                                   detector_backend='dlib')  # list of ('face', 'facial_area', 'confidence')
    crop_image = face_obj  # detectFace
    crop_image = crop_image * 255
    # crop_image = face_obj[0]['face']
    # crop_image = cv2.resize(crop_image, (224, 224))
    cv2.imwrite(out_path, crop_image[:, :, ::-1])


def main():
    # save_frames(r'./Hype04_short01.mp4', r'./Hype04_short01_frames')
    # check_face_detection(r'./Deepfake_01_short_01.jpg', './test_nicholas_0.jpg')
    # crop_face(r'/mnt/c/Projects/CTIC-Nicholas/Deepfake_01_short_01.mp4', r'/mnt/c/Projects/CTIC-Nicholas/nicholas_face_short', 512)

    video_dir = r'/data/CTIC_nicholas_raw_trimmed'
    face_dir = r'/data/CTIC_nicholas_face_frames'
    # video_list = ['Deepfake_01.mp4', 'Deepfake_02.mp4', 'Deepfake_03.mp4', 'Deepfake_04.mp4', 'Deepfake_05.mp4']
    video_list = ['Deepfake_06.mp4', 'Deepfake_07.mp4', 'Deepfake_08.mp4', 'Deepfake_09.mp4', 'Deepfake_10.mp4',
                  'Deepfake_11.mp4', 'Deepfake_12.mp4', 'Deepfake_13.mp4', 'Deepfake_14.mp4', 'Deepfake_15.mp4',
                  'Deepfake_16.mp4']

    for video_name in video_list:
        print(f'\nProcessing: {video_name}')
        video_path = os.path.join(video_dir, video_name)
        video_title = video_name.split('.')[0]
        face_path = os.path.join(face_dir, video_title)
        crop_face(video_path, face_path, 512, padding=50)
        # exit(1)


if __name__ == '__main__':
    main()