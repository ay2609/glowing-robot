import numpy as np

from glowing_robot.database import Profile


class Face:
    def __init__(self, box: np.ndarray, landmarks: np.ndarray, descriptor: np.ndarray) -> None:
        self._box = box
        self._landmarks: np.ndarray = landmarks
        self._descriptor = descriptor

        self._profile: Profile | None = None

    @property
    def box(self) -> np.ndarray:
        return self._box

    @property
    def landmarks(self) -> np.ndarray:
        return self._landmarks

    @property
    def descriptor(self) -> np.ndarray:
        return self._descriptor

    @property
    def profile(self) -> Profile | None:
        return self._profile
