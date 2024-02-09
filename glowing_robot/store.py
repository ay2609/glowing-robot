import numpy as np
from .database import add_descriptors, get_profile, save as save_database

from .model import model as model
from .face import Face


def image_to_faces(image: np.ndarray) -> list[Face]:
    boxes, probabilities, landmarks = model.detect(image)
    descriptors = model.compute_descriptors(image, boxes)

    faces = []

    for box, person_landmarks, descriptor in zip(boxes, landmarks, descriptors):
        faces.append(Face(box, person_landmarks, descriptor))

    return faces


def face_to_database(name: str, face: Face, save: bool = True) -> None:
    add_descriptors([name], [face.descriptor])
    face.profile = get_profile(name)

    if save:
        save_database()
