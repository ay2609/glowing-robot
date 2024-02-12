import numpy as np

from .face import Face
from .database import get_db
from .cosine_similarity import determine_similarity


def identify_face(face: Face):
    name_to_descriptor = np.array([(profile.mean, name) for name, profile in get_db().items])
    bool_to_similarity = determine_similarity(face.descriptor, name_to_descriptor[:, 0])

    best_descriptors = zip(name_to_descriptor[bool_to_similarity[:, 0]], bool_to_similarity[bool_to_similarity[:, 0]][:, 1])
    return best_descriptors
