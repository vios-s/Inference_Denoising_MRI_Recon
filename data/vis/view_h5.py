import numpy as np
import h5py
import os
from pathlib import Path
import sys
sys.path.append('../../')
import cv2 as cv
from natsort import natsorted

from utils.transform_utils import to_tensor, center_crop

data_path = Path('/home/s2287251/biomedic/Users/YYX/Projects/1st-year_Project/log/unet')

#lambda_ = ['0.3', '0.5', '0.7', '0.9']
#noise_lvl = ['0', '1e-8', '1e-6', '1e-4','2e-4', '4e-4', '6e-4', '8e-4', '1e-3', '3e-3']
lambda_ = ['0.1']
noise_lvl = ['1e-6']
for l in lambda_:
    for i in noise_lvl:
        h5_path = data_path.joinpath('lambda_' + l + '_noise_'+ i + '/')
        print(h5_path)
        for j in natsorted(h5_path.glob('*.h5')):
            dir_name = os.path.basename(j).split('.')[0]
            save_path = h5_path / dir_name
            if not save_path.exists():
                save_path.mkdir()

            with h5py.File(j, 'r') as hf:

                whole_image = hf["reconstruction"][()]
                for k in range(whole_image.shape[0]):
                    img_vis = np.asarray(whole_image[k] / np.max(whole_image[k]) * 255).astype(np.uint8)
                    print(img_vis.shape, img_vis.dtype)
                    img_name = str(save_path / '{}.png'.format(k))
                    print(img_name)
                    cv.imwrite(img_name, img_vis)
