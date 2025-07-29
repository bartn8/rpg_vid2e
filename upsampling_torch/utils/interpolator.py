"""The film_net frame interpolator main model code.

Basics
======
The film_net is an end-to-end learned neural frame interpolator implemented as
a PyTorch model. It has the following inputs and outputs:

Inputs:
  x0: image A.
  x1: image B.
  time: desired sub-frame time.

Outputs:
  image: the predicted in-between image at the chosen time in range [0, 1].

Additional outputs include forward and backward warped image pyramids, flow
pyramids, etc., that can be visualized for debugging and analysis.

Note that many training sets only contain triplets with ground truth at
time=0.5. If a model has been trained with such training set, it will only work
well for synthesizing frames at time=0.5. Such models can only generate more
in-between frames using recursion.

Architecture
============
The inference consists of three main stages: 1) feature extraction 2) warping
3) fusion. On high-level, the architecture has similarities to Context-aware
Synthesis for Video Frame Interpolation [1], but the exact architecture is
closer to Multi-view Image Fusion [2] with some modifications for the frame
interpolation use-case.

Feature extraction stage employs the cascaded multi-scale architecture described
in [2]. The advantage of this architecture is that coarse level flow prediction
can be learned from finer resolution image samples. This is especially useful
to avoid overfitting with moderately sized datasets.

The warping stage uses a residual flow prediction idea that is similar to
PWC-Net [3], Multi-view Image Fusion [2] and many others.

The fusion stage is similar to U-Net's decoder where the skip connections are
connected to warped image and feature pyramids. This is described in [2].

Implementation Conventions
====================
Pyramids
--------
Throughtout the model, all image and feature pyramids are stored as python lists
with finest level first followed by downscaled versions obtained by successively
halving the resolution. The depths of all pyramids are determined by
options.pyramid_levels. The only exception to this is internal to the feature
extractor, where smaller feature pyramids are temporarily constructed with depth
options.sub_levels.

Color ranges & gamma
--------------------
The model code makes no assumptions on whether the images are in gamma or
linearized space or what is the range of RGB color values. So a model can be
trained with different choices. This does not mean that all the choices lead to
similar results. In practice the model has been proven to work well with RGB
scale = [0,1] with gamma-space images (i.e. not linearized).

[1] Context-aware Synthesis for Video Frame Interpolation, Niklaus and Liu, 2018
[2] Multi-view Image Fusion, Trinidad et al, 2019
[3] PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
"""
from typing import Dict, List, Optional

import torch
from torch import nn

from .util import build_image_pyramid, flow_pyramid_synthesis, multiply_pyramid, concatenate_pyramids, pyramid_warp
from .feature_extractor import FeatureExtractor
from .fusion import Fusion
from .pyramid_flow_estimator import PyramidFlowEstimator
import numpy as np


def _pad_to_align(x: torch.Tensor, align: int):
    """Pad image batch x so width and height divide by align.

    Args:
        x: Image batch to align. Shape: [B, C, H, W]
        align: Number to align to.

    Returns:
        padded_x: Tensor padded so width % align == 0 and height % align == 0.
        bbox_to_crop: Dict with crop parameters to undo the padding.
    """
    assert x.ndim == 4
    assert align > 0, 'align must be a positive number.'

    _, _, height, width = x.shape
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    pad_top = height_to_pad // 2
    pad_bottom = height_to_pad - pad_top
    pad_left = width_to_pad // 2
    pad_right = width_to_pad - pad_left

    # torch.nn.functional.pad pads in (left, right, top, bottom) order for 4D tensors
    padded_x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    bbox_to_crop = {
        'offset_height': pad_top,
        'offset_width': pad_left,
        'target_height': height,
        'target_width': width
    }
    return padded_x, bbox_to_crop


class InterpolatorWrapper:
    """A class for generating interpolated frames between two input frames using PyTorch."""

    def __init__(self, model_path: str, align: Optional[int] = None) -> None:
        """Loads a saved PyTorch model.

        Args:
            model_path: Path to the saved model.
            align: If provided, pad the input size so it divides with this before inference.
        """
        # self._model = Interpolator()
        # self._model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # self._model.eval()

        self._model = torch.jit.load(model_path)
        self._model.eval()

        self._align = align if align is not None else None

    def to(self, device: torch.device):
        """Moves the model to the specified device."""
        self._model.to(device)
        return self

    def interpolate(self, x0: np.ndarray, x1: np.ndarray, dt: np.ndarray) -> tuple:
        """Generates an interpolated frame between given two batches of frames.

        All input tensors should be np.float32 datatype.

        Args:
            x0: First image batch. Dimensions: (batch_size, height, width, channels)
            x1: Second image batch. Dimensions: (batch_size, height, width, channels)
            dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

        Returns:
            A tuple containing:
            1) An image batch with interpolated frames. Shape: (batch_size, height, width, channels)
            2) Forward flow pyramid as a numpy array.
            3) Backward flow pyramid as a numpy array.
        """
        assert x0.ndim == 4 and x1.ndim == 4 and dt.ndim == 1, "Input dimensions are incorrect."
        assert x0.shape[0] == x1.shape[0] == dt.shape[0], "Batch sizes must match."
        
        device = next(self._model.parameters()).device

        # Convert numpy arrays to torch tensors and permute to [B, C, H, W]
        x0_torch = torch.from_numpy(x0).permute(0, 3, 1, 2).to(device)
        x1_torch = torch.from_numpy(x1).permute(0, 3, 1, 2).to(device)
        dt_torch = torch.from_numpy(dt).float().to(device)

        if self._align is not None:
            x0_torch, bbox_to_crop = _pad_to_align(x0_torch, self._align)
            x1_torch, _ = _pad_to_align(x1_torch, self._align)

        with torch.no_grad():
            result = self._model.debug_forward(x0_torch, x1_torch, dt_torch)
            image = result['image'][0]
            forward_flow = result['forward_flow_pyramid'][0].cpu().numpy()
            backward_flow = result['backward_flow_pyramid'][0].cpu().numpy()

        if self._align is not None:
            # Crop to original size
            offset_height = bbox_to_crop['offset_height']
            offset_width = bbox_to_crop['offset_width']
            target_height = bbox_to_crop['target_height']
            target_width = bbox_to_crop['target_width']
            image = image[:, :, offset_height:offset_height+target_height, offset_width:offset_width+target_width]

        # Convert back to numpy and [B, H, W, C]
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()
        return image_np, forward_flow, backward_flow

class Interpolator(nn.Module):
    def __init__(
            self,
            pyramid_levels=7,
            fusion_pyramid_levels=5,
            specialized_levels=3,
            sub_levels=4,
            filters=64,
            flow_convs=(3, 3, 3, 3),
            flow_filters=(32, 64, 128, 256),
    ):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels

        self.extract = FeatureExtractor(3, filters, sub_levels)
        self.predict_flow = PyramidFlowEstimator(filters, flow_convs, flow_filters)
        self.fuse = Fusion(sub_levels, specialized_levels, filters)

    def shuffle_images(self, x0, x1):
        return [
            build_image_pyramid(x0, self.pyramid_levels),
            build_image_pyramid(x1, self.pyramid_levels)
        ]

    def debug_forward(self, x0, x1, batch_dt) -> Dict[str, List[torch.Tensor]]:
        image_pyramids = self.shuffle_images(x0, x1)

        # Siamese feature pyramids:
        feature_pyramids = [self.extract(image_pyramids[0]), self.extract(image_pyramids[1])]

        # Predict forward flow.
        forward_residual_flow_pyramid = self.predict_flow(feature_pyramids[0], feature_pyramids[1])

        # Predict backward flow.
        backward_residual_flow_pyramid = self.predict_flow(feature_pyramids[1], feature_pyramids[0])

        # Concatenate features and images:

        # Note that we keep up to 'fusion_pyramid_levels' levels as only those
        # are used by the fusion module.

        forward_flow_pyramid = flow_pyramid_synthesis(forward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        backward_flow_pyramid = flow_pyramid_synthesis(backward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        # We multiply the flows with t and 1-t to warp to the desired fractional time.
        #
        # Note: In film_net we fix time to be 0.5, and recursively invoke the interpo-
        # lator for multi-frame interpolation. Below, we create a constant tensor of
        # shape [B]. We use the `time` tensor to infer the batch size.
        backward_flow = multiply_pyramid(backward_flow_pyramid, batch_dt)
        forward_flow = multiply_pyramid(forward_flow_pyramid, 1 - batch_dt)

        pyramids_to_warp = [
            concatenate_pyramids(image_pyramids[0][:self.fusion_pyramid_levels],
                                      feature_pyramids[0][:self.fusion_pyramid_levels]),
            concatenate_pyramids(image_pyramids[1][:self.fusion_pyramid_levels],
                                      feature_pyramids[1][:self.fusion_pyramid_levels])
        ]

        # Warp features and images using the flow. Note that we use backward warping
        # and backward flow is used to read from image 0 and forward flow from
        # image 1.
        forward_warped_pyramid = pyramid_warp(pyramids_to_warp[0], backward_flow)
        backward_warped_pyramid = pyramid_warp(pyramids_to_warp[1], forward_flow)

        aligned_pyramid = concatenate_pyramids(forward_warped_pyramid,
                                                    backward_warped_pyramid)
        aligned_pyramid = concatenate_pyramids(aligned_pyramid, backward_flow)
        aligned_pyramid = concatenate_pyramids(aligned_pyramid, forward_flow)

        return {
            'image': [self.fuse(aligned_pyramid)],
            'forward_residual_flow_pyramid': forward_residual_flow_pyramid,
            'backward_residual_flow_pyramid': backward_residual_flow_pyramid,
            'forward_flow_pyramid': forward_flow_pyramid,
            'backward_flow_pyramid': backward_flow_pyramid,
        }

    def forward(self, x0, x1, batch_dt) -> torch.Tensor:
        return self.debug_forward(x0, x1, batch_dt)['image'][0]
