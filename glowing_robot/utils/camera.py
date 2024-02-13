import cv2
import time
import numpy as np
from typing import Callable
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo


def get_available_cameras() -> list[str]:
    return [camera.localizedName() for camera in AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)]


def prompt_cameras() -> int:
    cameras = get_available_cameras()
    time.sleep(0.1)
    print("Available cameras:")
    for index, camera in enumerate(cameras):
        print(f'{index}) {camera}')

    got_input = False
    ind = None
    while not got_input:
        i_str = input('Enter the index: ')
        try:
            ind = int(i_str)
        except ValueError:
            print('Not a valid integer')
        else:
            if ind < 0 or ind >= len(cameras):
                print('Index out of range')
            else:
                got_input = True

    return ind


def stream_camera(camera_index: int, callback: Callable[[np.ndarray], bool], flip=True) -> None:
    print('Opening camera...')
    cap = cv2.VideoCapture(camera_index)
    try:
        print('Streaming...')
        last_ret = True
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to grab frame')
                if not last_ret:
                    break
            last_ret = ret
            if not ret:
                continue

            if flip:
                cv2.flip(frame, 1, frame)

            persist = callback(frame)
            if not persist:
                break
    finally:
        print('Closing camera...')
        cap.release()
