import os
from pathlib import Path
from typing import Optional, Union

from fractions import Fraction
from PIL import Image
import skvideo.io
import numpy as np

from .const import img_formats


class Sequence:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ImageSequence(Sequence):
    def __init__(self, imgs_dirpath: str, fps: Optional[float] = None):
        super().__init__()

        assert os.path.isdir(imgs_dirpath)
        self.imgs_dirpath = imgs_dirpath

        self.file_names = [f for f in os.listdir(imgs_dirpath) if self._is_img_file(f)]
        assert self.file_names
        self.file_names.sort()

        if fps is None:
            self.fps = len(self.file_names)
        else:
            assert fps > 0, 'Expected fps to be larger than 0. Instead got fps={}'.format(fps)
            self.fps = fps

    @classmethod
    def _is_img_file(cls, path: str):
        return Path(path).suffix.lower() in img_formats

    def __next__(self):
        for idx in range(0, len(self.file_names) - 1):
            file_paths = self._get_path_from_name([self.file_names[idx], self.file_names[idx + 1]])
            imgs = [self._pil_loader(f) for f in file_paths]
            times_sec = [idx/self.fps, (idx + 1)/self.fps]
            yield imgs, times_sec

    def __len__(self):
        return len(self.file_names) - 1

    @staticmethod
    def _pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

            w_orig, h_orig = img.size
            w, h = w_orig//32*32, h_orig//32*32

            left = (w_orig - w)//2
            upper = (h_orig - h)//2
            right = left + w
            lower = upper + h
            img = img.crop((left, upper, right, lower))
            return np.array(img).astype("float32") / 255

    def _get_path_from_name(self, file_names: Union[list, str]) -> Union[list, str]:
        if isinstance(file_names, list):
            return [os.path.join(self.imgs_dirpath, f) for f in file_names]
        return os.path.join(self.imgs_dirpath, file_names)

