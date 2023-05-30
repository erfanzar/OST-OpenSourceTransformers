import os

from flax.serialization import to_bytes, from_bytes, to_state_dict, from_state_dict
from typing import Union


def load_checkpoint(path: Union[str, os.PathLike], target=None, from_state_dict_: bool = False):
    with open(path, 'rb') as stream:
        data = from_bytes(target, stream.read())
    return data


def save_checkpoint(path: Union[str, os.PathLike], data, to_state_dict_: bool = False):
    with open(path, 'wb') as stream:
        data = to_bytes(data)
        stream.write(data)
    return path
