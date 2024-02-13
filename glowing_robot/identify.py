import numpy as np

from .face import Face
from .database import get_db, get_profile, Profile
from .cosine_similarity import determine_similarity


def identify_face(face: Face) -> None:
    names = np.array([name for name in get_db()])
    descriptors = np.array([get_profile(name).mean for name in names])
    bool_to_similarity = [determine_similarity(face.descriptor, descriptor) for descriptor in descriptors]
    bool_similarities = [bool_similarity for bool_similarity, _ in bool_to_similarity]
    similarities = np.array([similarity for _, similarity in bool_to_similarity])

    best_descriptors = dict(zip(names[bool_similarities], similarities[bool_similarities]))

    if len(best_descriptors) > 0:
        best_person = max(best_descriptors.items(), key=lambda x: x[1])
        best_profile = get_profile(best_person[0])
        face.profile = best_profile
