import numpy as np
from const import DESCRIPTOR_THRESHOLD


def _cosine_similarity(desc_1, desc_2):
    """
    :param desc_1: descriptor vector of face #1
    :param desc_2: descriptor vector of face #2
    :return: cosine_similarity: return a value between -1 to 1, -1 being opposite, 0 being unrelated, and 1 being similar
    """

    return np.dot(desc_1, desc_2) / (np.linalg.norm(desc_1) * np.linalg.norm(desc_2))


def determine_similarity(desc_1, desc_2, threshold=DESCRIPTOR_THRESHOLD) -> bool:
    """
    :param desc_1: descriptor vector of face #1
    :param desc_2: descriptor vector of face #2
    :param threshold: a float from -1 to 1 which determines what the threshold for finding two faces similar is
    :return: result: a boolean value which tells you if the two descriptor vectors (or faces) are similar
    """

    similarity = _cosine_similarity(desc_1, desc_2)

    return similarity > threshold, similarity
