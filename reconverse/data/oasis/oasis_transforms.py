import torch
import numpy as np
from typing import NamedTuple, Dict, Any, Optional, Callable
import sys
sys.path.append('../../../')

from reconverse.data.transform_utils import *
from reconverse.utils.math import complex_abs
from reconverse.utils.fftc import fft2c_new as fft2c, ifft2c_new as ifft2c


class OasisSample(NamedTuple):
    image: torch.Tensor
    target: torch.Tensor
    mean: float
    std: float
    fname: str
    slice_num: int
    metadata: dict
    max_value: float
    
    
class OasisDataTransform:
    def __init__(
        self,
        mask_func: Optional[Callable] = None,
        use_seed: bool = True
    ):
        self.mask_func = mask_func
        self.use_seed = use_seed
        
    def __call__(
        self,
        kspace: torch.Tensor,
        target: torch.Tensor,
        fname: str,
        slice_num: int,
        metadata: dict,
        mask: np.ndarray   
    ):
        """
        Args:
            kspace (np.ndarray): Input k-space of shape (num_coils, rows, cols) for multi-coil data
                or (rows, cols) for single coil data.
            mask (np.ndarray): Mask from the test dataset.
            target (np.ndarray): Target image.
            fname (str): File name.
            slice_num (int): slice index.

        Returns: A tuple containing,
            image: zero-filled input image,
            output: the reconstruction
            target: target,
            mean: the mean used for normalization,
            std: the standard deviations used for normalization,
            fname: the filename,
            slice_num: and the slice number.
        
        """
        max_value = 255.0
        # ! 1 apply mask
        if mask is None and self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask, _ = apply_mask(kspace, self.mask_func, seed=seed)
        else:
            masked_kspace = kspace * mask + 0.0
        
        # ! 2 get input images
        shift_masked_kspace = torch.fft.ifftshift(masked_kspace, dim=[-3, -2])
        input_masked_image = torch.view_as_real(
            torch.fft.ifftn(
                torch.view_as_complex(shift_masked_kspace), dim=(-2, -1), norm="ortho"
            )
        )
        input_masked_image = complex_abs(input_masked_image)
        
        # ! 3 normalize input image
        input_masked_image, mean, std = normalize_instance(input_masked_image, eps=1e-11)

        # ! 4 normalize target
        if target is not None:
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = complex_abs(normalize(target, mean, std, eps=1e-11))
            target = target.clamp(-6, 6)

        return OasisSample(
            image=input_masked_image.float(),
            target=target.float(),
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            metadata=metadata,
            max_value=max_value
        )