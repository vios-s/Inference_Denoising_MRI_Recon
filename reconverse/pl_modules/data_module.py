"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

from lightning.pytorch import LightningDataModule
import torch

import sys
sys.path.append('../../')

from reconverse.data.fastmri import CombinedSliceDataset, SliceDataset, VolumeSampler
from reconverse.data.oasis import OasisSliceDataset

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True

    return False


class OasisDataModule(LightningDataModule):
    def __init__(
            self,
            list_path: Path,
            data_path: Path,
            train_transform: Callable,
            val_transform: Callable,
            test_transform: Callable,
            test_path: Optional[Path] = None,
            sample_rate: Optional[float] = None,
            val_sample_rate: Optional[float] = None,
            test_sample_rate: Optional[float] = None,
            volume_sample_rate: Optional[float] = None,
            val_volume_sample_rate: Optional[float] = None,
            test_volume_sample_rate: Optional[float] = None,
            train_filter: Optional[Callable] = None,
            val_filter: Optional[Callable] = None,
            test_filter: Optional[Callable] = None,
            use_dataset_cache: bool = False,
            batch_size: int = 1,
            num_workers: int = 4,
    ):
        """_summary_

        Args:
            list_path (Path): Path to the csv file.
            data_path (Path): Path to root data directory. For example, if `knee/path`
                is the root directory with subdirectories `singlecoil_train` and
                `singlecoil_val`, you would input `knee/path` for data_path.
            challenge (str): Name of challenge from ('multicoil', 'singlecoil').
            train_transform (Callable): A transform object for the training dataset.
            val_transform (Callable): A transform object for the validation dataset.
            test_transform (Callable): A transform object for the test dataset.
            test_split (str, optional): Name of test split from ("test", "challenge").
                                        Defaults to "test".
            test_path (Optional[Path], optional):  An optional test path. Passing this overwrites
                                        data_path and test_split. Defaults to None.
            sample_rate (Optional[float], optional): Fraction of slices of the training data split to use.
                                        Can be set to less than 1.0 for rapid prototyping. If not set,
                                        it defaults to 1.0. To subsample the dataset either set
                                        sample_rate (sample by slice) or volume_sample_rate (sample by
                                        volume), but not both. Defaults to None.
            val_sample_rate (Optional[float], optional): Same as sample_rate, but for val split. Defaults to None.
            test_sample_rate (Optional[float], optional): Same as sample_rate, but for test split. Defaults to None.
            volume_sample_rate (Optional[float], optional): Same as sample rate but in volume. Defaults to None.
            val_volume_sample_rate (Optional[float], optional): Same as volume_sample_rate but for val split. Defaults to None.
            test_volume_sample_rate (Optional[float], optional): Same as volume_sample_rate but for test split. Defaults to None.
            train_filter (Optional[Callable], optional):  A callable which takes as input a training example
                                        metadata, and returns whether it should be part of the training
                                        dataset. Defaults to None.
            val_filter (Optional[Callable], optional): Same as train_filter but for val split. Defaults to None.
            test_filter (Optional[Callable], optional): Same as train_filter but for test split. Defaults to None.
            use_dataset_cache (bool, optional):  Whether to cache dataset metadata. This is
                                        very useful for large datasets like the brain data. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 1.
            num_workers (int, optional): Number of workers for PyTorch dataloader. Defaults to 4.
        """
        super().__init__()

        self.list_path = list_path
        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache = use_dataset_cache
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_data_loader(self,
                            data_transform: Callable,
                            data_partition: str,
                            sample_rate: Optional[float] = None) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            list_path = str(self.list_path) + "_train.csv"
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
        else:
            is_train = False
            if data_partition == "val":
                list_path = str(self.list_path) + "_val.csv"
                sample_rate = self.val_sample_rate if sample_rate is None else sample_rate
            elif data_partition == "test":
                list_path = str(self.list_path) + "_test.csv"
                sample_rate = self.test_sample_rate if sample_rate is None else sample_rate

        dataset = OasisSliceDataset(
            root=list_path,
            data_root=self.data_path,
            transform=data_transform,
            sample_rate=sample_rate
        )
        
        cdr_key = 'CDR'
        
        sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader
    
    def prepare_data(self):

            list_paths = [
                str(self.list_path) + "_train.csv",
                str(self.list_path) + "_val.csv",
                str(self.list_path) + "_test.csv"
            ]

            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform
            ]

            for i, (list_path, data_transform) in enumerate(zip(list_paths, data_transforms)):
                sample_rate = self.sample_rate
                _ = OasisSliceDataset(
                    root=list_path,
                    data_root=self.data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(self.test_transform, data_partition="test")
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--data_path", type=Path, default=None, help="Path to the root data directory")
        parser.add_argument("--test_path", type=Path, default=None,
                            help="Path to the test data directory, overwrites data-path and test_split.")
        parser.add_argument("--challenge", type=str, choices=("singlecoil", "multicoil"), default="singlecoil",
                            help="Which challenge")
        parser.add_argument("--test_split", type=str, choices=("test", "challenge"), default="test",
                            help="Which data partition to use as test split")
        parser.add_argument("--sample_rate", type=float, default=None, help="Fraction of slices to use for training")
        parser.add_argument("--val_sample_rate", type=float, default=None,
                            help="Fraction of slices to use for validation")
        parser.add_argument("--test_sample_rate", type=float, default=None,
                            help="Fraction of slices to use for testing")
        parser.add_argument("--volume_sample_rate", type=float, default=None,
                            help="Fraction of volumes to use for training")
        parser.add_argument("--val_volume_sample_rate", type=float, default=None,
                            help="Fraction of volumes to use for validation")
        parser.add_argument("--test_volume_sample_rate", type=float, default=None,
                            help="Fraction of volumes to use for testing")
        parser.add_argument("--use_dataset_cache", type=bool, default=True,
                            help="Whether to cache dataset metadata in memory")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

        return parser
    

class FastMriDataModule(LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            sample_rate: Fraction of slices of the training data split to use.
                Can be set to less than 1.0 for rapid prototyping. If not set,
                it defaults to 1.0. To subsample the dataset either set
                sample_rate (sample by slice) or volume_sample_rate (sample by
                volume), but not both.
            val_sample_rate: Same as sample_rate, but for val split.
            test_sample_rate: Same as sample_rate, but for test split.
            volume_sample_rate: Fraction of volumes of the training data split
                to use. Can be set to less than 1.0 for rapid prototyping. If
                not set, it defaults to 1.0. To subsample the dataset either
                set sample_rate (sample by slice) or volume_sample_rate (sample
                by volume), but not both.
            val_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            test_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            train_filter: A callable which takes as input a training example
                metadata, and returns whether it should be part of the training
                dataset.
            val_filter: Same as train_filter, but for val split.
            test_filter: Same as train_filter, but for test split.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError(
                "Can set val_sample_rate or val_volume_sample_rate, but not both."
            )
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError(
                "Can set test_sample_rate or test_volume_sample_rate, but not both."
            )

        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.val_filter
            elif data_partition == "test":
                sample_rate = (
                    self.test_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.test_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.test_filter

        # if desired, combine train and val together for the train split
        dataset: Union[SliceDataset, CombinedSliceDataset]
        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
            ]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / f"{self.challenge}_{data_partition}"

            dataset = SliceDataset(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
            )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None

        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / f"{self.challenge}_test"
            data_paths = [
                self.data_path / f"{self.challenge}_train",
                self.data_path / f"{self.challenge}_val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                # NOTE: Fixed so that val and test use correct sample rates
                sample_rate = self.sample_rate  # if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                _ = SliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform, data_partition=self.test_split
        )

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--test_split",
            choices=("val", "test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (train split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--val_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (val split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--test_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (test split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (train split only). If not "
                "given all will be used. Cannot set together with sample_rate."
            ),
        )
        parser.add_argument(
            "--val_volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (val split only). If not "
                "given all will be used. Cannot set together with val_sample_rate."
            ),
        )
        parser.add_argument(
            "--test_volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (test split only). If not "
                "given all will be used. Cannot set together with test_sample_rate."
            ),
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=bool,
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser