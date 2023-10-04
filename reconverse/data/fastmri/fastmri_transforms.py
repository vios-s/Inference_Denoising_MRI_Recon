"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import sys
sys.path.append('../../../')

from reconverse.data.transform_utils import *
from reconverse.utils.math import complex_abs
from reconverse.utils.coil_combine import rss, rss_complex
from reconverse.utils.fftc import fft2c_new as fft2c, ifft2c_new as ifft2c


class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value: float


class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_torch

        # inverse Fourier transform to get zero filled solution
        image = ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)

        # absolute value
        image = complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            target_torch = center_crop(target_torch, crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])

        return UnetSample(
            image=image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )


class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> VarNetSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = VarNetSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )
        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = VarNetSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )

        return sample


class MiniCoilSample(NamedTuple):
    """
    A sample of masked coil-compressed k-space for reconstruction.

    Args:
        kspace: the original k-space before masking.
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    kspace: torch.Tensor
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]


class MiniCoilTransform:
    """
    Multi-coil compressed transform, for faster prototyping.
    """

    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        use_seed: Optional[bool] = True,
        crop_size: Optional[tuple] = None,
        num_compressed_coils: Optional[int] = None,
    ):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            crop_size: Image dimensions for mini MR images.
            num_compressed_coils: Number of coils to output from coil
                compression.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.crop_size = crop_size
        self.num_compressed_coils = num_compressed_coils

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset. Not used if mask_func is defined.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                kspace: original kspace (used for active acquisition only).
                masked_kspace: k-space after applying sampling mask. If there
                    is no mask or mask_func, returns same as kspace.
                mask: The applied sampling mask
                target: The target image (if applicable). The target is built
                    from the RSS opp of all coils pre-compression.
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        if target is not None:
            target = to_tensor(target)
            max_value = attrs["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        if self.crop_size is None:
            crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])
        else:
            if isinstance(self.crop_size, tuple) or isinstance(self.crop_size, list):
                assert len(self.crop_size) == 2
                if self.crop_size[0] is None or self.crop_size[1] is None:
                    crop_size = torch.tensor(
                        [attrs["recon_size"][0], attrs["recon_size"][1]]
                    )
                else:
                    crop_size = torch.tensor(self.crop_size)
            elif isinstance(self.crop_size, int):
                crop_size = torch.tensor((self.crop_size, self.crop_size))
            else:
                raise ValueError(
                    f"`crop_size` should be None, tuple, list, or int, not: {type(self.crop_size)}"
                )

        if self.num_compressed_coils is None:
            num_compressed_coils = kspace.shape[0]
        else:
            num_compressed_coils = self.num_compressed_coils

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = 0
        acq_end = crop_size[1]

        # new cropping section
        square_crop = (attrs["recon_size"][0], attrs["recon_size"][1])
        kspace = fft2c(
            complex_center_crop(ifft2c(to_tensor(kspace)), square_crop)
        ).numpy()
        kspace = complex_center_crop(kspace, crop_size)

        # we calculate the target before coil compression. This causes the mini
        # simulation to be one where we have a 15-coil, low-resolution image
        # and our reconstructor has an SVD coil approximation. This is a little
        # bit more realistic than doing the target after SVD compression
        target = rss_complex(ifft2c(to_tensor(kspace)))
        max_value = target.max()

        # apply coil compression
        new_shape = (num_compressed_coils,) + kspace.shape[1:]
        kspace = np.reshape(kspace, (kspace.shape[0], -1))
        left_vec, _, _ = np.linalg.svd(kspace, compute_uv=True, full_matrices=False)
        kspace = np.reshape(
            np.array(np.matrix(left_vec[:, :num_compressed_coils]).H @ kspace),
            new_shape,
        )
        kspace = to_tensor(kspace)

        # Mask kspace
        if self.mask_func:
            masked_kspace, mask, _ = apply_mask(
                kspace, self.mask_func, seed, (acq_start, acq_end)
            )
            mask = mask.byte()
        elif mask is not None:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask = mask.byte()
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]

        return MiniCoilSample(
            kspace, masked_kspace, mask, target, fname, slice_num, max_value, crop_size
        )