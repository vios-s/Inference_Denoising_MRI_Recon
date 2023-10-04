"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from lightning.pytorch import LightningModule
import torch
from torchmetrics.metric import Metric

import sys
sys.path.append('../../')
from reconverse.utils import evaluate
from reconverse.utils.io import save_reconstructions


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16, recon_dir: str = "reconstruction"):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.recon_dir = recon_dir
        self.val_log_indices = None
        self.train_log_indices = [0]
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.TrainLoss = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TestLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def on_validation_epoch_end(self):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in self.validation_step_outputs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"].keys():
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = metrics["psnr"] + 20 * torch.log10(torch.tensor(max_vals[fname], dtype=mse_val.dtype, device=mse_val.device)) - 10 * torch.log10(mse_val)
            metrics["ssim"] = metrics["ssim"] + torch.mean(torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()]))


        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)
            
        self.validation_step_outputs.clear()


    def on_test_epoch_end(self):
        outputs = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in self.test_step_outputs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructions to {save_path}")

        save_reconstructions(outputs, save_path)
        self.test_step_outputs.clear()


    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument("--num_log_images", default=16, type=int, help="Number of images to log to Tensorboard")
        parser.add_argument("--recon_dir", default="reconstruction", type=str)


        return parser