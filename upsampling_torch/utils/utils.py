import os
from pathlib import Path
from typing import Union
import glob

from .dataset import Sequence, ImageSequence
from .const import img_formats


def get_imgs_directory(dirpath: str) -> Union[None, str]:
    # check if the directory contains images
    for ext in img_formats:
        imgs = glob.glob(os.path.join(dirpath, f"*{ext}"))
        if imgs:
            return dirpath
    return None

def get_sequence_or_none(dirpath: str) -> Union[None, Sequence]:
    imgs_dir = get_imgs_directory(dirpath)
    if imgs_dir:
        return ImageSequence(imgs_dir, None)
    return None


