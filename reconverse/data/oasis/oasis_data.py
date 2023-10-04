from pathlib import Path
from typing import NamedTuple, Dict, Any, Optional, Callable

import os
import csv
import random
import torch
import nibabel as nib
import numpy as np
from glob import glob


class OasisDataSample(NamedTuple):
    """
    Basic data type for OASIS raw data.

    Elements:
        fname: id for each patient, Path
        slice_ind: slice index, int
        metadata: metadata for each volume, Dict
    """
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]
    
    
class OasisSliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: list,
        data_root: list,
        transform: Optional[Callable] = None,
        sample_rate: Optional[float] = None,
    ):
        self.root = root
        self.data_root = data_root
        self.transform = transform
        self.sample_rate = sample_rate
        self.raw_samples = self.create_sample_list(self.root)
        
    def __getitem__(self, index):
        fname, dataslice, metadata = self.raw_samples[index]
        raw_target = self.read_raw_data(self.data_root, fname, dataslice)
        _, raw_kspace = self.im2kp(torch.tensor(raw_target))
        if self.transform is None:
            sample = (raw_kspace, torch.tensor(raw_target), fname, dataslice, metadata)
        else:
            sample = self.transform(raw_kspace, torch.tensor(raw_target), fname, dataslice, metadata, mask=None)
            
        return sample
    
    def __len__(self):
        return len(self.raw_samples)
    
    def read_raw_data(self, data_path, patient_id, dataslice):
        image_path = glob(os.path.join(data_path, patient_id, 'PROCESSED', 'MPRAGE', 'T88_111', '*t88_gfc.img'))
        image_data = nib.load(image_path[0]).get_fdata()
        raw_image = image_data.squeeze(-1)
        raw_slices = raw_image[dataslice]
        
        return raw_slices
        
    def create_sample_list(self, list_path):
        raw_samples = []
        with open(list_path) as metalist:
            datalist = csv.reader(metalist, delimiter=',')
            next(datalist)
            for row in datalist:
                fname = row[0]
                metadata = {
                    'CDR': row[1],
                }
                # We only take axial slices, change to other dimension if you want
                slices = 175
                margin = round(slices/4)
                for num in range(margin, slices-margin):
                    new_raw_sample = (OasisDataSample(fname, num, metadata))
                    raw_samples.append(new_raw_sample)
            metalist.close()
            
        if self.sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * self.sample_rate)
            raw_samples = self.raw_samples[:num_raw_samples]
            
        return raw_samples
    
    def im2kp(self,image_data, norm_type="ortho"):
        # perform fft in last two dimensions of input data
        kspace_complex_data = torch.fft.fft2(image_data, dim=(-2, -1), norm=norm_type)
        kspace_split_data = torch.view_as_real(kspace_complex_data)
        kspace = torch.fft.fftshift(kspace_split_data, dim=(-3, -2))

        return kspace_complex_data, kspace

    
    
                    
        
        
    
    