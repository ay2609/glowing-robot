from threading import Thread, Lock

import cv2
import numpy as np

from glowing_robot import image_to_faces, face_to_database, prompt_cameras, stream_camera, Face, list_entries, identify_face

cam_index = 0
# cam_index = prompt_cameras()
faces: list[Face] = []
proc_done = True
face_lock = Lock()


def get_color(index: int) -> tuple[str, tuple[int, int, int]]:
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (225, 0, 255)]
    names = ['Red', 'Green', 'Blue', 'Yellow']

    index = index % len(names)

    return names[index], colors[index]


def prompt_new_name(check=True) -> str:
    while True:
        name = input('Enter the new name: ').strip()
        if check:
            if name not in list_entries():
                print('Confirm that this is a new person: ')
                text = input(': ')
                if not text.lower().startswith('y'):
                    continue
        return name


def print_faces(colors: list[tuple[str, tuple[int, int, int]]]) -> None:
    global faces
    for index, face, color in zip(range(len(faces)), faces, colors):
        name = 'Unknown'
        if face.profile is not None:
            name = face.profile.name
        print(f' {index}) {name} ({color[0]})')


def prompt_faces(colors: list[tuple[str, tuple[int, int, int]]], frame: np.ndarray) -> None:
    global faces
    while True:
        print('Select a person:')
        print_faces(colors)
        print('Or anything else to resume.')
        text = input(': ')

        try:
            index = int(text)
            face = faces[index]
        except (ValueError, IndexError):
            break
        else:
            known = face.profile is not None
            name = 'Unknown'
            if known:
                name = face.profile.name

            print(f'Select an option for {name}:')
            if known:
                print(' 0) Add to db (Correct)')
                print(' 1) Rename')
            else:
                print(' 0) Add to db')
            print('Anything else to cancel')
            text = input(': ').strip()

            if text == '0':
                if known:
                    face_to_database(name, face)
                else:
                    face_to_database(prompt_new_name(), face)
                face.profile.last_image = frame
                print('Added')
            elif known and text == '1':
                face_to_database(prompt_new_name(False), face)
                face.profile.last_image = frame
                print('Renamed')


def prompt_faces_last_image(colors: list[tuple[str, tuple[int, int, int]]]) -> None:
    global faces
    while True:
        print('Select a person:')
        print_faces(colors)
        print('Or anything else to resume.')
        text = input(': ')

        try:
            index = int(text)
            face = faces[index]
        except (ValueError, IndexError):
            break
        else:
            if face.profile is not None and face.profile.last_image is not None:
                print('Press space to resume')
                window = f'Last image of {face.profile.name}'
                cv2.imshow(window, face.profile.last_image)
                while True:
                    if cv2.waitKey(1) == ord(' '):
                        break
                cv2.destroyWindow(window)
                cv2.waitKey(1)
                break
            else:
                print('That face is not known')


def update_faces(frame: np.ndarray) -> None:
    global faces, proc_done, face_lock
    temp_faces = image_to_faces(frame)
    if not face_lock.locked():
        with face_lock:
            faces = temp_faces
    proc_done = True


def process_frame(frame: np.ndarray) -> bool:
    global faces, proc_done, face_lock
    og_frame = frame.copy()
    for face in faces:
        face: Face
        identify_face(face)
        known = face.profile is not None

        center: tuple = tuple(np.clip(face.landmarks.astype(np.uint32)[2], [0, 0], frame.shape[:-1][::-1]))
        box = np.clip(face.box.astype(np.uint32), [0, 0, 0, 0], list(frame.shape[:-1][::-1]) * 2)
        width = min(abs((box[2] - box[0]) // 2), frame.shape[1])
        height = min(abs((box[3] - box[1]) // 2), frame.shape[0])

        cv2.ellipse(frame, center, (width, height), 0, 0, 360, (0, 255, 0) if known else (0, 0, 255), thickness=3)

        if known:
            org = (min(abs(box[0]), frame.shape[1]), min(abs(box[3]), frame.shape[0]))
            cv2.putText(frame, face.profile.name.title(), org, cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

    if proc_done:
        proc_done = False
        Thread(target=update_faces, args=[frame], daemon=True).start()

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key in (ord(' '), ord('\t')):
        with face_lock:
            colors = []
            frame = og_frame.copy()
            for index, face in zip(range(len(faces)), faces):
                face: Face
                known = face.profile is not None

                colors.append(get_color(index))
                color = colors[-1][1]

                center: tuple = tuple(np.clip(face.landmarks.astype(np.uint32)[2], [0, 0], frame.shape[:-1][::-1]))
                box = np.clip(face.box.astype(np.uint32), [0, 0, 0, 0], list(frame.shape[:-1][::-1]) * 2)
                width = min(abs((box[2] - box[0]) // 2), frame.shape[1])
                height = min(abs((box[3] - box[1]) // 2), frame.shape[0])

                cv2.ellipse(frame, center, (width, height), 0, 0, 360, color, thickness=3)

                if known:
                    org = (min(abs(box[0]), frame.shape[1]), min(abs(box[3]), frame.shape[0]))
                    cv2.putText(frame, face.profile.name.title(), org, cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)

            if key == ord(' '):
                prompt_faces(colors, og_frame)
            else:
                prompt_faces_last_image(colors)

            cv2.waitKey(1)
    elif key == ord('q'):
        return False

    return True


def main() -> None:
    stream_camera(cam_index, process_frame)


if __name__ == "__main__":
    main()
