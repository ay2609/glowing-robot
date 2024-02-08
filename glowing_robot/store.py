import numpy as np

from model import model as model
from face import Face


def image_to_faces(image: np.ndarray) -> list[Face]:
    boxes, probabilities, landmarks = model.detect(image)
    descriptors = model.compute_descriptors(image, boxes)

    faces = []

    for box, person_landmarks, descriptor in zip(boxes, landmarks, descriptors):
        faces.append(Face(box, person_landmarks, descriptor))

    return faces
