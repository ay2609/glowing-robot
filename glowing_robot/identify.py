import numpy as np

from .face import Face
from .database import get_db, get_profile, Profile
from .cosine_similarity import determine_similarity


def identify_face(face: Face) -> None:
    name_to_descriptor = [(profile.mean, name) for name, profile in get_db().items() if len(profile.mean) > 0]
    bool_to_similarity = determine_similarity(face.descriptor, np.array(name_to_descriptor)[:, 0])

    best_descriptors = zip(name_to_descriptor[bool_to_similarity[:, 1]], bool_to_similarity[bool_to_similarity[:, 0]][:, 1])

    if not best_descriptors:
        best_person = max(best_descriptors, key=lambda x: x[1])
        best_profile = get_profile(best_person[0])
        face.profile = best_profile
        best_profile.add_descriptors(best_person[1])
        face.profile = best_profile
