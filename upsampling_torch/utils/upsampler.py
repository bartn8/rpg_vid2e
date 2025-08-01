import os
import shutil
from typing import Optional
import torch
import cv2
import numpy as np
from tqdm import tqdm

from . import Sequence
from .interpolator import InterpolatorWrapper
from .utils import get_sequence_or_none


class Upsampler:
    _timestamps_filename = 'timestamps.txt'

    def __init__(self, input_dir: str, output_dir: str, pretrained_model_path: Optional[str] = None, max_bisections: int = 5):
        assert os.path.isdir(input_dir), 'The input directory must exist'

        if os.path.exists(output_dir):
            # Remove the existing output directory if it exists (also files)
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.src_dir = input_dir
        self.dest_dir = output_dir
        self.max_bisections = max_bisections

        path = os.path.join(os.path.dirname(__file__), "../../pretrained_models/film_net_fp32.pt") if pretrained_model_path is None else pretrained_model_path
        self.interpolator = InterpolatorWrapper(path, None)
        self.interpolator.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def upsample(self):
        
        seq_list = []

        for src_absdirpath, dirnames, filenames in os.walk(self.src_dir):
            sequence = get_sequence_or_none(src_absdirpath)
            if sequence is None:
                continue
            seq_list.append(src_absdirpath)

        #sort the sequences by their absolute path
        seq_list.sort()


        for src_absdirpath in tqdm(seq_list, desc="Processing sequences"):
            sequence = get_sequence_or_none(src_absdirpath)
            if sequence is None:
                continue

            reldirpath = os.path.relpath(src_absdirpath, self.src_dir)
            dest_imgs_dir = os.path.join(self.dest_dir, reldirpath)
            dest_timestamps_filepath = os.path.join(self.dest_dir, reldirpath, "..", self._timestamps_filename)
            self.upsample_sequence(sequence, dest_imgs_dir, dest_timestamps_filepath)

    def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
        os.makedirs(dest_imgs_dir, exist_ok=True)
        timestamps_list = list()

        idx = 0
        for img_pair, time_pair in tqdm(next(sequence), total=len(sequence), desc=type(sequence).__name__):
            I0 = img_pair[0][None]
            I1 = img_pair[1][None]
            t0, t1 = time_pair

            total_frames, total_timestamps = self._upsample_adaptive(I0, I1, t0, t1)
            total_frames = [I0[0]] + total_frames
            timestamps = [t0] + total_timestamps

            sorted_indices = np.argsort(timestamps)
            total_frames = [total_frames[j] for j in sorted_indices]
            timestamps = [timestamps[i] for i in sorted_indices]

            timestamps_list += timestamps
            for frame in total_frames:
                self._write_img(frame, idx, dest_imgs_dir)
                idx += 1

        timestamps_list.append(t1)
        self._write_img(I1[0, ...], idx, dest_imgs_dir)
        self._write_timestamps(timestamps_list, dest_timestamps_filepath)

    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections=-1):
        if num_bisections == 0:
            return [], []

        dt = self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        image, F_0_1, F_1_0 = self.interpolator.interpolate(I0, I1, dt)

        if num_bisections < 0:
            flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** .5).max()
            flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** .5).max()

            # Calculate the number of bisections based on the maximum flow magnitude
            num_bisections = int(np.ceil(np.log(max([flow_mag_0_1_max, flow_mag_1_0_max]))/np.log(2)))
            num_bisections = min(num_bisections, self.max_bisections)
            # print(f"Number of bisections: {num_bisections}")

            if num_bisections == 0:
                return [image[0]], [(t0 + t1) / 2]

        left_images, left_timestamps = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
        right_images, right_timestamps = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)
        timestamps = left_timestamps + [(t0+t1)/2] + right_timestamps
        images = left_images + [image[0]] + right_images

        return images, timestamps

    @staticmethod
    def _write_img(img: np.ndarray, idx: int, imgs_dir: str):
        assert os.path.isdir(imgs_dir)
        img = np.clip(img * 255, 0, 255).astype("uint8")
        path = os.path.join(imgs_dir, "%08d.jpg" % idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)

    @staticmethod
    def _write_timestamps(timestamps: list, timestamps_filename: str):
        np.savetxt(timestamps_filename, (np.array(timestamps)*1e9).astype(np.int64), fmt='%d', delimiter='\n')
