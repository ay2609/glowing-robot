import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TypeVar, Union

import numpy as np
import skimage.io as io
from facenet_models import FacenetModel

_T = TypeVar("_T")

_default_path = Path(os.path.dirname(os.path.abspath(__file__))) / "face_db.pkl"
_path = _default_path


class Profile:
    """ Saves profile information in the face-recognition database."""

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            '<first-name> <last-name>' used to identify the database profile.
        """
        self.name = name
        try:
            self.first, self.last = name.split()
        except TypeError:
            raise TypeError(
                "A face-database profile name must have a first and last name: 'John Smith' "
            )
        self._descriptors: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None

    def add_descriptors(self, descriptors: np.ndarray):
        """Add descriptors to profile.
        Parameters
        ----------
        descriptors : numpy.ndarray
            One, shape=(D,), or more, shape=(N, D), descriptor arrays.
        """
        if self._descriptors is None:
            if descriptors.ndim == 1:
                descriptors = descriptors[np.newaxis, :]
            self._descriptors = descriptors
        else:
            self._descriptors = np.vstack([self._descriptors, descriptors])
        self._mean = self._descriptors.mean(axis=0)

    @property
    def num_entries(self) -> int:
        """Returns the number of descriptors stored for this profile.
        Returns
        -------
        int
        """
        return 0 if self._descriptors is None else len(self._descriptors)

    @property
    def mean(self) -> np.ndarray:
        """The average descriptor for this profile.
        Returns
        -------
        numpy.ndarray, shape=(D,)
        """
        return self._mean


_face_db: Optional[Dict[str, Profile]] = None


def _load(force: bool = False):
    """Load the database from face_rec/face_db/face_db.pkl if it isn't
    already loaded.
    Call this if you want to load the database up front. Otherwise,
    the other database methods will automatically load it.
    """
    global _face_db
    if _face_db is not None and not force:
        return None

    if not _path.is_file():
        print(
            f"No face-database found. Creating empty database...\n\tSaving it will save to {_path.absolute()}"
        )
        _face_db = dict()
    else:
        with _path.open(mode="rb") as f:
            _face_db = pickle.load(f)
        print(f"face-database loaded from: {_path.absolute()}")


def load_face_db(func: Optional[_T] = None) -> _T:
    """This function can be invoked directly to lazy-load the face-recognition database, or it can
    be used as a decorator: the database is lazy-loaded prior to invoking the decorated function.
    See face_rec.face_db._load for more information.
    Parameters
    ----------
    func : Optional[Callable]
    Returns
    -------
    Union[None, Callable]
    """
    if func is None:
        _load()
        return None

    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        _load()
        return func(*args, **kwargs)

    return wrapper


def save():
    """ Save the database."""
    if _face_db is None:
        print("No changes to face-database to save")
        return None

    with _path.open(mode="wb") as f:
        pickle.dump(_face_db, f)

    print(f"Face database saved to: {_path.absolute()}")


def switch_db(path: Optional[Union[str, _Path]] = None):
    """Switch the face database being used by specifying its load/save path. Calling this
    function with no argument will revert to the default database.
    Providing a name with no directories will assume face_rec/face_db as the directory,
    otherwise the provided path is used. All databases will be saved as .pkl files.
    Parameters
    ----------
    path : PathLike
    """
    from pathlib import Path

    global _face_db
    _backup_db = _face_db

    global _path
    _backup_path = _path

    try:
        if path is not None:
            path = Path(path)
            parent = path.parent if str(path.parent) != "." else _default_path.parent
            _path = parent / (path.stem + ".pkl")
        else:
            _path = _default_path
        _face_db = None
        load_face_db()
    except Exception as e:
        print(f"The following error occurred: {e}")
        print(f"\nReverting to your prior database state at: {_path.absolute()}")
        _face_db = _backup_db
        _path = _backup_path


@load_face_db
def add_images(
        name: str,
        items: Union[str, _Path, np.ndarray, Iterable[Union[str, _Path, np.ndarray]]],
):
    """Extract face-descriptors from the provided images, and add them to the
    face-recognition database.
    A new profile will be created if `name.lower()` is not in the database.
    Parameters
    ----------
    name : str
        '<first-name> <last-name>' used to identify the database profile.
    items : Union[PathLike, numpy.ndarray, Sequence[Union[PathLike, numpy.ndarray]]]
        One or more paths to images, or RGB-valued numpy arrays, from which the descriptors
        are extracted. Each image must contain exactly one face.
    """

    try:
        _, _ = name.split()
    except TypeError:
        raise TypeError(
            "A face-database profile name must have a first and last name: 'John Smith' "
        )

    if isinstance(items, (str, _Path)) or (
            isinstance(items, np.ndarray) and items.ndim == 3
    ):
        items = [items]

    def to_3_channel(x):
        return x[..., :-1] if x.shape[-1] == 4 else x  # png -> RGB

    arrays = (
        to_3_channel(io.imread(x)) if isinstance(x, (str, _Path)) else x for x in items
    )

    descriptors = []
    model = FacenetModel()

    for n, array in enumerate(arrays):
        boxes, _, _ = model.detect(array)

        # each picture should contain one face
        if len(boxes) != 1:
            print(f"Warning: item {n} contains a picture with {len(boxes)} faces")
            print("This item was skipped.. each item should contain exactly 1 face")
            continue

        descriptors.append(model.compute_descriptors(array, boxes))

    if descriptors:
        name = name.lower()
        profile = _face_db.setdefault(name, Profile(name))
        profile.add_descriptors(np.vstack(descriptors))
        print(
            f"{name} had {len(descriptors)} descriptors added to his/her profile; {profile.num_entries} in total"
        )


@load_face_db
def add_descriptors(names: Iterable[str], descriptors: Iterable[np.ndarray]):
    """Add descriptors for multiple database profiles.
    Parameters
    ----------
    names : Iterable[str]
        N names.
    descriptors : Sequence[numpy.ndarray]
        A sequence of N descriptors/descriptor-blocks. That is,
        each element of the sequence can be a (512,) array or
        a (M, 512) array, corresponding to M descriptors for that
        profile.
    """
    for name, descriptor in zip(names, descriptors):
        if name is None:
            continue
        name = name.lower()
        profile = _face_db.setdefault(name, Profile(name))
        profile.add_descriptors(descriptor)


@load_face_db
def list_entries() -> List[str]:
    """Returns a list of alphabetized profile-names in the database.
    Returns
    -------
    List[str]
    """
    return sorted(_face_db.keys())


@load_face_db
def get_profile(key: str):
    """Returns the profile-instance for the specified profile-name from the database.
    Parameters
    ----------
    key : str
        '<first-name> <last-name>' used to identify the database entry.
    Returns
    -------
    face_rec.face_db._Profile
    """
    return _face_db[key.lower()]


@load_face_db
def delete_profile(key: str):
    """Removes the specified profile from the database.
    Parameters
    ----------
    key : str
        '<first-name> <last-name>' used to identify the database entry.
    """
    _face_db.pop(key)
