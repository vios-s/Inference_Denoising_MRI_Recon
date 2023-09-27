"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import xml.etree.ElementTree as etree
from argparse import ArgumentParser
from pathlib import Path

import h5py
from tqdm import tqdm


from data.fastmri.fastmri_data import et_query
from data.fastmri import fastmri_transforms

from utils.fftc import ifft2c_new as ifft2c
from utils.math import complex_abs
from utils.coil_combine import rss
from utils.io import save_reconstructions

def save_zero_filled(data_dir, out_dir, which_challenge):
    reconstructions = {}

    for fname in tqdm(list(data_dir.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            masked_kspace = fastmri_transforms.to_tensor(hf["kspace"][()])

            # extract target image width, height from ismrmrd header
            enc = ["encoding", "encodedSpace", "matrixSize"]
            crop_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
            )

            # inverse Fourier Transform to get zero filled solution
            image = ifft2c(masked_kspace)

            # check for FLAIR 203
            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            # crop input image
            image = fastmri_transforms.complex_center_crop(image, crop_size)

            # absolute value
            image = complex_abs(image)

            # apply Root-Sum-of-Squares if multicoil data
            if which_challenge == "multicoil":
                image = rss(image, dim=1)

            reconstructions[fname.name] = image

    save_reconstructions(reconstructions, out_dir)


def create_arg_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the reconstructions to",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        required=True,
        help="Which challenge",
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    save_zero_filled(args.data_path, args.output_path, args.challenge)